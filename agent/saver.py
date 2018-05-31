import os
import torch as th
from .agent import Agent


class Saver(object):
    def __init__(self, module: Agent, model_checkpoint_path: str):
        self.module = module
        self.path = False
        if os.path.exists(model_checkpoint_path + '/checkpoint'):
            self.model_checkpoint_path = model_checkpoint_path
            self.path = True
        else:
            if not os.path.exists(model_checkpoint_path):
                os.makedirs(model_checkpoint_path)
            self.model_checkpoint_path = model_checkpoint_path
            self._write(0)
            self._save(0)
            wall_t_fname = self.model_checkpoint_path + '/' + 'wall_t.0'
            with open(wall_t_fname, 'w') as f:
                f.write('0')

    def restore(self):
        filename = self._read()
        start_epoch = self._restore(filename)
        return start_epoch

    def _restore(self, filename):
        path = self.model_checkpoint_path + '/' + filename
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = th.load(path)
        start_epoch = checkpoint['epoch']
        self.module.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {})"
              .format(start_epoch + 1))
        return start_epoch

    def save(self, epoch):
        filename = self._read()
        self._remove(filename)
        self._write(epoch)
        self._save(epoch)

    def _save(self, epoch):
        path = self.model_checkpoint_path + '/' + '{}-model.tar'.format(epoch)
        state = {
            'epoch': epoch,
            'state_dict': self.module.state_dict(),
        }
        th.save(state, path)

    def _read(self):
        with open(self.model_checkpoint_path + '/checkpoint', 'r', encoding='UTF-8') as checkpoint:
            filename = checkpoint.read().split(':')[1]
        return filename

    def _write(self, epoch):
        with open(self.model_checkpoint_path + '/checkpoint', 'w', encoding='UTF-8') as checkpoint:
            checkpoint.write('epoch {}:{}-model.tar'.format(epoch, epoch))

    def _remove(self, filename):
        os.remove(self.model_checkpoint_path + '/' + filename)
