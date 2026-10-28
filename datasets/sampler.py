from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (pid, _, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length



class HardNegativeIdentitySampler(Sampler):
    """
    Identity sampler whose batches are made of *look-alike* identities.

    Every batch has P = batch_size // num_instances identities with `num_instances` samples each.
    The identities are a random seed plus P-1 identities drawn from the seed's `pool` nearest
    identities (nearest = most confusable under the backbone's global text-image similarity,
    see utils/hard_batch.py). Inside such a batch the hardest in-batch negatives are as hard as the
    candidates the reranker meets at test time (the 16 hardest of the whole gallery), instead of
    being easy ones drawn from a random batch.

    Until set_neighbors() is called the neighbour lists are empty and the batches are random
    identity batches. The sampler yields len(self) indices per epoch (a multiple of batch_size).
    """

    def __init__(self, data_source, batch_size, num_instances, pool=30):
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = max(1, batch_size // num_instances)
        self.pool = pool
        self.index_dic = defaultdict(list)
        for index, (pid, _, _, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.neighbors = {}
        n_batches = max(1, len(data_source) // batch_size)
        self.length = n_batches * batch_size

    def set_neighbors(self, neighbors):
        """neighbors: dict pid -> list of the most similar *other* pids (nearest first)."""
        self.neighbors = {pid: list(nb[:self.pool]) for pid, nb in neighbors.items()}

    def _pick_pids(self):
        P = self.num_pids_per_batch
        seed = random.choice(self.pids)
        chosen = [seed]
        nb = [p for p in self.neighbors.get(seed, []) if p in self.index_dic and p != seed]
        if nb:
            chosen += random.sample(nb, min(P - 1, len(nb)))
        if len(chosen) < P:  # no / too few neighbours known: fill with random identities
            taken = set(chosen)
            while len(chosen) < P and len(taken) < len(self.pids):
                p = random.choice(self.pids)
                if p not in taken:
                    taken.add(p)
                    chosen.append(p)
        return chosen

    def __iter__(self):
        out = []
        for _ in range(self.length // self.batch_size):
            batch = []
            for pid in self._pick_pids():
                idxs = self.index_dic[pid]
                if len(idxs) >= self.num_instances:
                    batch.extend(random.sample(idxs, self.num_instances))
                else:
                    batch.extend(random.choices(idxs, k=self.num_instances))
            # P * num_instances can be smaller than batch_size when batch_size % num_instances != 0
            while len(batch) < self.batch_size:
                batch.append(random.choice(self.index_dic[random.choice(self.pids)]))
            out.extend(batch[:self.batch_size])
        return iter(out)

    def __len__(self):
        return self.length
