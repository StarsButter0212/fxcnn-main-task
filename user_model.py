import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.normal import Normal

import copy
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime
from fxcnn.min_norm_solvers import MinNormSolver, gradient_normalizers

# --------------------------------- Shared-Bottom -----------------------------------------
class BottomNet(nn.Module):
    def __init__(self, ninp, nhid, ndepths):
        super(BottomNet, self).__init__()

        layers = []
        for i in range(ndepths):
            n1 = ninp if i == 0 else nhid
            layers.append(nn.Linear(n1, nhid))
            layers.append(nn.Softplus())
        self.shared_net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.shared_net(x)
        return x

    def get_last_shared_layer(self):
        return self.shared_net[2 * (self.ndepths-1)]

class TowerNet(nn.Module):
    def __init__(self, ninp, nhid, ndepths=1, noutp=1):
        super(TowerNet, self).__init__()

        layers = []
        for i in range(ndepths):
            if layers:
                layers.append(nn.Softplus())
            n1 = ninp if i == 0 else nhid
            n2 = noutp if i == (ndepths - 1) else nhid
            layers.append(nn.Linear(n1, n2))
        self.tower_net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.tower_net(x)
        return x

class Shared_Bottom(nn.Module):
    def __init__(self, input_size, bottom_hidden, bottom_depths, towers_hidden):
        super(Shared_Bottom, self).__init__()

        self.shared = BottomNet(input_size, bottom_hidden, bottom_depths)
        self.tasks_ae = TowerNet(bottom_hidden, towers_hidden, ndepths=1)
        self.tasks_ie = TowerNet(bottom_hidden, towers_hidden, ndepths=1)
        self.tasks_dens = TowerNet(bottom_hidden, towers_hidden, ndepths=1)

    def forward(self, x, t):
        x = self.shared(x)
        x = eval('self.tasks_{}'.format(t))(x)
        return x


# --------------------------------- MoE simple -----------------------------------------
class GatingFunction(nn.Module):
    def __init__(self, input_size, hidden_size, num_experts):
        super(GatingFunction, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_experts)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.softmax(x, dim=1)
        return x

class Expert(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

class MixtureOfExperts(nn.Module):
    def __init__(self, input_size_expert, hidden_size_expert, output_size,
                 num_experts, input_size_gating=None, hidden_size_gating=None):
        super(MixtureOfExperts, self).__init__()

        if input_size_gating is None:
            input_size_gating = input_size_expert
        if hidden_size_gating is None:
            hidden_size_gating = hidden_size_expert
        self.gating_function = GatingFunction(input_size_gating,
                                              hidden_size_gating, num_experts)
        self.experts = nn.ModuleList(
            [Expert(input_size_expert, hidden_size_expert, output_size)
             for _ in range(num_experts)])

    def forward(self, x_expert, x_gating=None):
        if x_gating is None:
            x_gating = x_expert
        gate_output = self.gating_function(x_gating)

        expert_outputs = [expert(x_expert) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        output = torch.sum(gate_output.unsqueeze(-1) * expert_outputs, dim=1)
        return output


# --------------------------------- Trainer and tester -----------------------------------------
class Trainer(object):
    def __init__(self, model, hparams):
        self.model = model
        self._hparams = hparams
        self.initial_task_loss = None

        opt_str = self._hparams.get("optimizer", "adam").lower()
        if opt_str == "adam":
            opt_cls = optim.Adam
        elif opt_str == "radam":
            opt_cls = optim.RAdam
        elif opt_str == "adamw":
            opt_cls = optim.AdamW
        else:
            raise RuntimeError("Unknown optimizer %s" % opt_str)

        lr, wdecay = self._hparams["lr"], self._hparams.get("wdecay", 0.0)
        iteration = self._hparams["max_epochs"]

        if self._hparams["multi_tasks"]:
            params = self.model.parameters()
            self.optimizer = opt_cls(params, lr=lr, weight_decay=wdecay)
            self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, 5*lr,
                                                           total_steps=iteration, pct_start=0.2)
        else:
            # different optimizers and schedulers for each type data
            params = list(self.model.parameters())
            self.optimizers = [opt_cls(params, lr=self._hparams["%slr" % tpe], weight_decay=wdecay)
                               for tpe in self.model.weights]
            self.schedulers = [torch.optim.lr_scheduler.OneCycleLR(self.optimizers[i],  # called every epoch automatically
                                                         5*self._hparams["%slr" % tpe], total_steps=iteration,
                                                         pct_start=0.2) for i, tpe in enumerate(self.model.weights)]

    def optimize(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def train(self, dataloader, epoch):
        self.model.train()

        if self._hparams['user_model'] and self._hparams["multi_tasks"]:
            """Minimize two loss functions in terms of E."""
            loss_task = {}
            loss_data = {}
            grads = {}
            scale = {}
            losses, task_losses = 0, 0
            tasks = list(set(self._hparams['tasks']) - set(self._hparams['exclude_types']))
            tasks.sort(key=self._hparams['tasks'].index)

            dataloaders = [dataloader[t] for t in tasks]
            loop_data = tqdm(enumerate(zip(*dataloaders)), total=len(dataloaders[0]))
            start_time = datetime.now()

            for index, train_batch in loop_data:
                train_batch = dict(zip(tasks, train_batch))
                # Scaling the loss functions based on the algorithm choice
                if 'mgda' in self._hparams['algorithm']:
                    for t in tasks:
                        # Comptue gradients of each loss function wrt parameters
                        self.optimizer.zero_grad()
                        loss_t, _ = self.model(train_batch[t], t)
                        loss_task[t] = loss_t.item()
                        loss_t.backward()
                        grads[t] = []
                        for param in self.model.nnmodel.shared.parameters():
                            if param.grad is not None:
                                grads[t].append(Variable(param.grad.data.clone(), requires_grad=False))
                        # print('{} grads: {}'.format(t, grads[t]))

                    # Normalize all gradients, this is optional and not included in the paper.
                    gn = gradient_normalizers(grads, loss_task, self._hparams['normalization_type'])
                    for t in tasks:
                        for gr_i in range(len(grads[t])):
                            grads[t][gr_i] = grads[t][gr_i] / gn[t]

                    # Frank-Wolfe iteration to compute scales.
                    solver = MinNormSolver()
                    sol, min_norm = solver.find_min_norm_element([grads[t] for t in tasks])
                    for i, t in enumerate(tasks):
                        scale[t] = float(sol[i])
                else:
                    for t in tasks:
                        scale[t] = self.model.weights[t]

                # Scaled back-propagation
                for i, t in enumerate(tasks):
                    loss_t, _ = self.model(train_batch[t], t)
                    loss_data.setdefault(t, []).append(loss_t.item())
                    if i > 0:
                        loss = loss + scale[t] * loss_t
                    else:
                        loss = scale[t] * loss_t

                self.optimize(loss, self.optimizer)
                losses += loss.item()

                delta_time = datetime.now() - start_time
                loop_data.set_description('\33[36m【Epoch {0:04d}】'.format(epoch))
                loop_data.set_postfix({'loss': '{0:.6f}'.format(losses / (index + 1)),
                                       'loss_ae': '{0:.6f}'.format(np.mean(loss_data['ae']) * scale['ae']),
                                       'loss_ie': '{0:.6f}'.format(np.mean(loss_data['ie']) * scale['ie']),
                                       'loss_dens': '{0:.6f}'.format(np.mean(loss_data['dens']) * scale['dens']),
                                       'cost_time': '{0}'.format(delta_time)}, '\33[0m')

            self.scheduler.step()
            return losses/len(dataloaders[0]), np.mean(loss_data['ae']), \
                   np.mean(loss_data['ie']), np.mean(loss_data['dens'])
        else:
            """Minimize two loss functions in terms of E."""
            losses = 0
            loss_ae, loss_ie, loss_dens = 0, 0, 0
            losses_ae, losses_ie, losses_dens = [], [], []
            loop_data = tqdm(enumerate(dataloader), total=len(dataloader))
            start_time = datetime.now()

            for index, train_batch in loop_data:
                tpe = train_batch["type"]
                if self._hparams["split_opt"]:
                    idx = self.model.type_indices[tpe]
                else:
                    idx = 0
                opt = self.optimizers[idx]
                sche = self.schedulers[idx]

                loss, _ = self.model(train_batch)
                self.optimize(loss, opt)
                losses += loss.item()

                if tpe == 'ae':
                    losses_ae += [loss.item()]
                    loss_ae = np.mean(losses_ae)
                elif tpe == 'ie':
                    losses_ie += [loss.item()]
                    loss_ie = np.mean(losses_ie)
                elif tpe == 'dens':
                    losses_dens += [loss.item()]
                    loss_dens = np.mean(losses_dens)
                else:
                    print('Error: Wrong type training !')

                delta_time = datetime.now() - start_time
                loop_data.set_description('\33[36m【Epoch {0:04d}】'.format(epoch))
                loop_data.set_postfix({'loss': '{0:.6f}'.format(losses / (index + 1)),
                                       'loss_ae': '{0:.6f}'.format(loss_ae),
                                       'loss_ie': '{0:.6f}'.format(loss_ie),
                                       'loss_dens': '{0:.6f}'.format(loss_dens),
                                       'cost_time': '{0}'.format(delta_time)}, '\33[0m')

            sche.step()
            return losses/len(dataloader), loss_ae, loss_ie, loss_dens

    def get_gradient(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward(retain_graph=True)

    def set_gradient(self, grads, optimizer, shapes):
        for group in optimizer.param_groups:
            length = 0
            for i, p in enumerate(group['params'][:len(shapes)]):
                i_size = np.prod(shapes[i])
                get_grad = grads[length:length + i_size]
                length += i_size
                p.grad = get_grad.view(shapes[i])

    def pcgrad_fn(self, model, losses, optimizer, mode='mean'):
        grad_list = {}
        shapes = []
        shares = []
        for i, tpe in enumerate(losses):
            self.get_gradient(losses[tpe], optimizer)
            grads = []
            for p in model.nnmodel.shared.parameters():
                if i == 0:
                    shapes.append(p.shape)
                if p.grad is not None:
                    grads.append(p.grad.view(-1))
                else:
                    grads.append(torch.zeros_like(p).view(-1))
            new_grad = torch.cat(grads, dim=0)
            grad_list[tpe] = new_grad

            if shares == []:
                shares = (new_grad != 0)
            else:
                shares &= (new_grad != 0)

        # clear memory
        loss_all = sum(losses.values())
        loss_all.backward()

        print()
        grad_list2 = copy.deepcopy(grad_list)
        for g_i in grad_list:
            key_list2 = list(grad_list2.keys())
            random.shuffle(key_list2)
            for g_j in key_list2:
                g_i_g_j = torch.dot(grad_list[g_i], grad_list2[g_j])
                if g_i_g_j < 0:
                    print('conflicting tasks:', g_i, g_j)
                    grad_list[g_i] -= (g_i_g_j) * grad_list2[g_j] / (grad_list2[g_j].norm() ** 2)

        grads = torch.cat(list(grad_list.values()), dim=0)
        grads = grads.view(len(losses), -1)

        if mode == 'mean':
            grads_share = grads * shares.float()
            grads_share = grads_share.mean(dim=0)
            grads_no_share = grads * (1 - shares.float())
            grads_no_share = grads_no_share.sum(dim=0)

            grads = grads_share + grads_no_share
        else:
            grads = grads.sum(dim=0)

        self.set_gradient(grads, optimizer, shapes)

        return loss_all.item()



class Tester(object):
    def __init__(self, model, hparams):
        self.model = model
        self._hparams = hparams

    def test(self, dataloader):
        self.model.eval()
        losses = 0
        MAE_ae, DEV_ie, DEV_dens = 0, 0, 0
        MAEs_ae, DEVs_ie, DEVs_dens = [], [], []
        losses_ae, losses_ie, losses_dens = [], [], []

        for index, valid_batch in enumerate(dataloader):
            with torch.no_grad():
                task = valid_batch["type"]
                if self._hparams['user_model'] and self._hparams["multi_tasks"]:
                    loss, deviation = self.model(valid_batch, task)
                else:
                    loss, deviation = self.model(valid_batch)

                losses += loss.item()
                losses_val = losses / (index + 1)

                if task == 'ae':
                    losses_ae += [loss.item()]
                    MAEs_ae += [deviation.item()]
                    loss_ae, MAE_ae = np.mean(losses_ae), np.mean(MAEs_ae)
                elif task == 'ie':
                    losses_ie += [loss.item()]
                    DEVs_ie += [deviation.item()]
                    loss_ie, DEV_ie = np.mean(losses_ie), np.mean(DEVs_ie)
                elif task == 'dens':
                    losses_dens += [loss.item()]
                    DEVs_dens += [deviation.item()]
                    loss_dens, DEV_dens = np.mean(losses_dens), np.mean(DEVs_dens)
                else:
                    print('Error: Wrong type testing !')

        return losses_val, MAE_ae, DEV_ie, DEV_dens

    def save_result(self, result, filename):
        with open(filename, 'a') as f:
            f.write(result + '\n')

    def save_prediction(self, prediction, filename):
        with open(filename, 'w') as f:
            f.write(prediction)

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)
