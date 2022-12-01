from Utils import ops, utils
from Utils.ops import *
from torch.backends import cudnn

cudnn.benchmark = True


class MyCell(nn.Module):
    def __init__(self, genotype, c_previous, c_current):
        super(MyCell, self).__init__()
        self.preprocess0 = ops.StdConv(c_previous, c_current, 1, 1, 0)
        self.genotype = genotype
        self.C = c_current
        self.concat = []
        self.op_names = []
        self.indices = []
        self._compile()
        self._steps = len(self.op_names) // 2

    def parse(self):
        count = 0
        for hidden in self.genotype:
            count += 1
            for i in hidden[0]:
                self.indices.append(i)
            for j in hidden[2]:
                self.op_names.append(j)

            if hidden[1] != 0:
                self.concat.append(count)

    def _compile(self):
        self.parse()
        assert len(self.op_names) == len(self.indices)
        self.multiplier = len(self.concat) if len(self.concat) >= 2 else 1
        self._ops = nn.ModuleList()
        stride = 1
        for op_name, index in zip(self.op_names, self.indices):
            op = ops.MyOps[op_name](self.C, stride, True)
            self._ops += [op]

    def forward(self, s0):
        s0 = self.preprocess0(s0)
        states = [s0]
        drop_prob = 0.2
        h = [None] * 10
        op = [None] * 10
        for i in range(self._steps):
            h[0] = states[self.indices[2 * i]]
            h[1] = states[self.indices[2 * i + 1]]
            op[0] = self._ops[2 * i]
            op[1] = self._ops[2 * i + 1]
            h[0] = op[0](h[0])
            h[1] = op[1](h[1])

            # if self.training and drop_prob > 0.:
            #     if not isinstance(op[0], Identity):
            #         h[0] = utils.drop_path(h[0], drop_prob)
            #     if not isinstance(op[1], Identity):
            #         h[1] = utils.drop_path(h[1], drop_prob)

            s = h[0] + h[1]

            states += [s]
        out = torch.cat([states[i] for i in self.concat], dim=1)
        return out


class Network(nn.Module):
    def __init__(self, C, num_classes, N, genotype):
        super(Network, self).__init__()
        self._N = N
        self._auxiliary = False

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev, C_curr = C_curr, C
        self.cells = nn.ModuleList()
        channel_concate_num = 5
        for i in range(0, self._N):
            cell = MyCell(genotype[0], C_prev, C_curr)
            self.cells += [cell]
            C_prev = C_curr * channel_concate_num
        self.cells += [nn.MaxPool2d(kernel_size=2, stride=2)]
        C_curr *= 2
        for i in range(0, self._N):
            cell = MyCell(genotype[1], C_prev, C_curr)
            self.cells += [cell]
            C_prev = C_curr * channel_concate_num
        self.cells += [nn.MaxPool2d(kernel_size=2, stride=2)]
        C_curr *= 2
        for i in range(0, self._N):
            cell = MyCell(genotype[2], C_prev, C_curr)
            self.cells += [cell]
            C_prev = C_curr * channel_concate_num

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        s0 = self.stem(input)
        for cell in self.cells:
            s0 = cell(s0)
        out = self.global_pooling(s0)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits


def train(train_loader, valid_loader, model, w_optim, lr, epoch, total_epoch):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch * len(train_loader)

    model.train()

    for step, (trn_x, trn_y) in enumerate(train_loader):
        trn_x, trn_y = trn_x.cuda(non_blocking=True), trn_y.cuda(non_blocking=True)
        N = trn_x.size(0)

        w_optim.zero_grad()
        logits = model(trn_x)
        criteria = nn.CrossEntropyLoss()
        loss = criteria(logits, trn_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        w_optim.step()

        prec1, prec5 = utils.accuracy(logits, trn_y, top_k=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % 50 == 0 or step == len(train_loader) - 1:
            print(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch + 1, total_epoch, step, len(train_loader) - 1, losses=losses,
                    top1=top1, top5=top5))

        cur_step += 1


def validate(valid_loader, model, epoch, cur_step, total_epoch):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (x, y) in enumerate(valid_loader):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            N = x.size(0)

            logits = model(x)
            criteria = nn.CrossEntropyLoss()
            loss = criteria(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, top_k=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % 50 == 0 or step == len(valid_loader) - 1:
                print(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch + 1, total_epoch, step, len(valid_loader) - 1, losses=losses,
                        top1=top1, top5=top5))

    return top1.avg


def evaluate_genotype(genotype, C, num_classes, N, total_epochs, lr, batch_size):
    pass


def evaluate_indi_during_search(genotype, C, num_classes, N, total_epochs, lr, batch_size):
    pass