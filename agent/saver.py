import os
import torch as th


class Saver(object):
    def __init__(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = os.path.join(path, "checkpoint.tar")

    def restore(self, model):
        if not os.path.exists(self.path):
            print("Could not find old checkpoint")
            return 0, 0
        print("=> loading checkpoint")
        checkpoint = th.load(self.path)
        start_epoch = checkpoint['epoch']
        wall_time = checkpoint['wall_time']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint (epoch {} wall time {})".format(start_epoch + 1, wall_time))
        return start_epoch, wall_time

    def save(self, model, epoch, wall_time):
        state = {
            'epoch': epoch,
            'wall_time': wall_time,
            'state_dict': model.state_dict(),
        }
        th.save(state, self.path)
