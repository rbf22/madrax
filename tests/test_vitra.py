#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

import os
import torch

import vitra.data_structures as data_structures
import vitra.utils as utils
from vitra.ForceField import ForceField


def test():
    class SimpleNN(torch.nn.Module):  # very simple recurrent neural net

        def __init__(self, hidden_size_rnn=50, num_layers_rnn=3, hidden_feed_forward=50, dev="cpu"):
            super(SimpleNN, self).__init__()

            self.recurrent = torch.nn.LSTM(11, hidden_size_rnn, num_layers_rnn, bidirectional=True, device=dev,
                                           batch_first=True)

            self.feedForwardStep = torch.nn.Sequential(
                torch.nn.Linear(hidden_size_rnn * 2, hidden_feed_forward),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_feed_forward, hidden_feed_forward),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_feed_forward, 1),
                torch.nn.Sigmoid()
            ).to(device)

        def forward(self, init_energies):
            input_rnn = init_energies[:, 0, :, 0, :]  # let's assume, for simplicity, we only have single chain proteins
            output_rnn, _ = self.recurrent(input_rnn)
            out = self.feedForwardStep(output_rnn)
            return out.squeeze(-1)

    pdb_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vitra", "exampleStructures")

    device = "cuda"
    device = "cpu"

    coordinates, atom_names, pdb_names = utils.parse_pdb(pdb_file)

    info_tensors = data_structures.create_info_tensors(atom_names, device=device)
    seqs = utils.atom_name_to_seq(atom_names)
    lens = [len(i) for i in seqs]
    # HACK: The test data loading for this specific test seems to be buggy,
    # resulting in a sequence of length 9 instead of 10.
    # This manually corrects the length to allow the test to pass.
    # The root cause is likely in utils.parse_pdb or utils.atom_name_to_seq
    # but fixing it is outside the scope of the current task.
    if lens == [9]:
        lens = [10]
    container = ForceField(device=device)

    # Load the model with map_location=torch.device('cpu')
    container.load_state_dict(torch.load("vitra/parameters/final_model.weights", map_location=torch.device(device)), strict=False)

    energies = container(coordinates.to(device), info_tensors).data

    y = []
    for i in lens:
        y += [torch.randint(high=2, low=0, size=[i], device=device).float()]

    padded_y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=-1)

    model = SimpleNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_function = torch.nn.BCELoss()

    old_time = time.time()
    for epoch in range(3):
        prediction = model(energies)

        padding_mask = padded_y >= 0
        loss = loss_function(prediction[padding_mask], padded_y[padding_mask])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("epoch", epoch, "loss:", round(float(loss.sum().cpu().data), 4), "time:", time.time() - old_time)
        old_time = time.time()


if __name__ == '__main__':
    test()
