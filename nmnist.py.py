import argparse
import os
from time import time as t
from bindsnet.network.network import Network

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import tonic

from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages
from bindsnet.learning import PostPre
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder

import numpy as np

import os

from bindsnet import ROOT_DIR
from bindsnet.analysis.plotting import (
    plot_assignments,
    plot_input,
    plot_performance,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--n_train", type=int, default=60000)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--update_steps", type=int, default=256)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--nu1", type=float, default=1e-2)
parser.add_argument("--nu2", type=float, default=1e-5)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--norm", type=float, default=1)
parser.add_argument("--time", type=int, default=100)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.set_defaults(plot=True, gpu=True)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
batch_size = args.batch_size
n_epochs = args.n_epochs
n_test = args.n_test
n_train = args.n_train
n_workers = args.n_workers
update_steps = args.update_steps
exc = args.exc
inh = args.inh
nu1 = args.nu1
nu2 = args.nu2
norm = args.norm
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
train = args.train
plot = args.plot
gpu = args.gpu

update_interval = update_steps * batch_size
per_class = int(n_neurons / 10)

device = "cpu"
# Sets up Gpu use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False

torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

# Determines number of workers to use
if n_workers == -1:
    n_workers = 0  # gpu * 1 * torch.cuda.device_count()

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity

# Build network.
if True:
    # best_norm = 2 * 34 * 34/10
    # best_nu1 = 1e-5
    # best_nu2 = 1e-2
    # best_exc = 22.5
    # best_inh = 120
    network = DiehlAndCook2015(
        n_inpt=2 * 34 * 34,
        n_neurons=n_neurons,
        exc=exc,
        inh=inh,
        dt=dt,
        norm=norm,
        nu=(nu1, nu2),
        theta_plus=theta_plus,
        inpt_shape=(2, 34, 34),
    )

    # Voltage recording for excitatory and inhibitory layers.
    exc_voltage_monitor = Monitor(
        network.layers["Ae"], ["v"], time=int(time / dt), device=device
    )
    inh_voltage_monitor = Monitor(
        network.layers["Ai"], ["v"], time=int(time / dt), device=device
    )
    network.add_monitor(exc_voltage_monitor, name="exc_voltage")
    network.add_monitor(inh_voltage_monitor, name="inh_voltage")

else:
    network = Network()
    inpt = Input(2 * 34 * 34, shape=(2, 34, 34), traces=True,sum_input = True)
    network.add_layer(inpt, name="Ae")
   
    output_layer = LIFNodes(10, thresh=-52 + np.random.randn(10).astype(float),traces=True,sum_input = True)
    network.add_layer(output_layer, name="Ai")

    C2 = Connection(source=inpt, target=output_layer, norm=150,wmin=-1,wmax=1)
    network.add_connection(C2, source="Ae", target="Ai")


# Directs network to GPU
if gpu:
    network.to("cuda")


dataset_path = "/content/data"



# Load MNIST data.
sensor_size = tonic.datasets.NMNIST.sensor_size
transform = transforms.Compose([tonic.transforms.Denoise(filter_time=10000),
                                tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=time),])
testset = tonic.datasets.NMNIST(save_to=dataset_path,
                                train=False,
                                transform=transform)
# testloader = DataLoader(testset, batch_size=1, shuffle=True)


trainset = tonic.datasets.NMNIST(save_to=dataset_path,
                                train=True,
                                transform=transform)
# trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

# Neuron assignments and spike proportions.
n_classes = 10
assignments = -torch.ones(n_neurons, device=device)
proportions = torch.zeros((n_neurons, n_classes), device=device)
rates = torch.zeros((n_neurons, n_classes), device=device)

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}


# Set up monitors for spikes and voltages
spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(
        network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
    )
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"X"}:
    voltages[layer] = Monitor(
        network.layers[layer], state_vars=["v"], time=int(time / dt), device=device
    )
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes, voltage_ims = None, None

spike_record = torch.zeros((update_interval, int(time / dt), n_neurons), device=device)

# Train the network.
print("\nBegin training.\n")
start = t()

for epoch in range(n_epochs):
    labels = []

    if epoch % progress_interval == 0:
        print("\n Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    # Create a dataloader to iterate and batch data
    train_dataloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=gpu,
    )

    pbar_training = tqdm(total=n_train)
    count = 0
    for step, (events,batch_labels) in enumerate(train_dataloader):

        if step > n_train:
            break
        # Get next input sample.
        inputs = {"X": events.view(time, batch_size, 2, 34, 34)}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        if step % update_steps == 0 and step > 0:
            # Convert the array of labels into a tensor
            label_tensor = torch.tensor(labels, device=device)

            # Get network predictions.
            all_activity_pred = all_activity(
                spikes=spike_record, assignments=assignments, n_labels=n_classes
            )
            proportion_pred = proportion_weighting(
                spikes=spike_record,
                assignments=assignments,
                proportions=proportions,
                n_labels=n_classes,
            )

            # Compute network accuracy according to available classification strategies.
            accuracy["all"].append(
                100
                * torch.sum(label_tensor.long() == all_activity_pred).item()
                / len(label_tensor)
            )
            accuracy["proportion"].append(
                100
                * torch.sum(label_tensor.long() == proportion_pred).item()
                / len(label_tensor)
            )

            print(
                "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
                % (
                    accuracy["all"][-1],
                    np.mean(accuracy["all"]),
                    np.max(accuracy["all"]),
                )
            )
            print(
                "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f"
                " (best)\n"
                % (
                    accuracy["proportion"][-1],
                    np.mean(accuracy["proportion"]),
                    np.max(accuracy["proportion"]),
                )
            )

            # Assign labels to excitatory layer neurons.
            assignments, proportions, rates = assign_labels(
                spikes=spike_record,
                labels=label_tensor,
                n_labels=n_classes,
                rates=rates,
            )

            labels = []

        labels.extend(batch_labels.tolist())


        # Run the network on the input.
        n_clamp = 1 # ???
        choice = np.random.choice(int(n_neurons / n_classes), size=n_clamp, replace=False)
        clamp = {"Ae": per_class * batch_labels.long() + torch.Tensor(choice).long()}

        network.run(inputs=inputs, time=time, clamp=clamp)

        # Add to spikes recording.
        s = spikes["Ae"].get("s").permute((1, 0, 2))
        spike_record[
            (step * batch_size)
            % update_interval : (step * batch_size % update_interval)
            + s.size(0)
        ] = s



        # Get voltage recording.
        exc_voltages = exc_voltage_monitor.get("v")
        inh_voltages = inh_voltage_monitor.get("v")

        # Optionally plot various simulation information.
        if False:
            image = batch["image"][:, 0].view(28, 28)
            inpt = inputs["X"][:, 0].view(time, 784).sum(0).view(28, 28)
            lable = batch["label"][0]
            input_exc_weights = network.connections[("X", "Ae")].w
            square_weights = get_square_weights(
                input_exc_weights.view(784, n_neurons), n_sqrt, 28
            )
            square_assignments = get_square_assignments(assignments, n_sqrt)
            spikes_ = {
                layer: spikes[layer].get("s")[:, 0].contiguous() for layer in spikes
            }
            voltages = {"Ae": exc_voltages, "Ai": inh_voltages}
            inpt_axes, inpt_ims = plot_input(
                image, inpt, label=lable, axes=inpt_axes, ims=inpt_ims
            )
            spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
            weights_im = plot_weights(square_weights, im=weights_im)
            assigns_im = plot_assignments(square_assignments, im=assigns_im)
            perf_ax = plot_performance(
                accuracy, x_scale=update_steps * batch_size, ax=perf_ax
            )
            voltage_ims, voltage_axes = plot_voltages(
                voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
            )

            plt.pause(1e-8)

        network.reset_state_variables()  # Reset state variables.
        pbar_training.update(batch_size)

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Training complete.\n")

# Load MNIST data.
# test_dataset = MNIST(
#     PoissonEncoder(time=time, dt=dt),
#     None,
#     root=os.path.join(ROOT_DIR, "data", "MNIST"),
#     download=True,
#     train=False,
#     transform=transforms.Compose(
#         [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
#     ),
# )

# Create a dataloader to iterate and batch data
test_dataloader = DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=n_workers,
    pin_memory=gpu,
)

# Sequence of accuracy estimates.
accuracy = {"all": 0, "proportion": 0}

# Train the network.
print("\nBegin testing\n")
network.train(mode=False)
start = t()

pbar = tqdm(total=n_test)
for step, (events,batch_labels) in enumerate(test_dataloader):
    if step > n_test:
        break
    # Get next input sample.
    inputs = {"X": events.view(time, batch_size, 2, 34, 34)}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run the network on the input.
    network.run(inputs=inputs, time=time, input_time_dim=1)

    # Add to spikes recording.
    spike_record = spikes["Ae"].get("s").permute((1, 0, 2))

    # Convert the array of labels into a tensor
    label_tensor = torch.tensor(batch_labels, device=device)

    # Get network predictions.
    all_activity_pred = all_activity(
        spikes=spike_record, assignments=assignments, n_labels=n_classes
    )
    proportion_pred = proportion_weighting(
        spikes=spike_record,
        assignments=assignments,
        proportions=proportions,
        n_labels=n_classes,
    )

    # Compute network accuracy according to available classification strategies.
    accuracy["all"] += float(torch.sum(label_tensor.long() == all_activity_pred).item())
    accuracy["proportion"] += float(
        torch.sum(label_tensor.long() == proportion_pred).item()
    )

    network.reset_state_variables()  # Reset state variables.
    pbar.set_description_str("Test progress: ")
    pbar.update()

print("\nAll activity accuracy: %.2f" % (accuracy["all"] / n_test))
print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / n_test))

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Testing complete.\n")

import csv
f = open("/content/hyperparameter_data.csv","a")
spamwriter = csv.writer(f)
test_acc = (accuracy["all"] / n_test)
print(test_acc)

new_row = [test_acc, np.mean(accuracy["all"]), n_neurons,exc]
new_row += [inh,nu1,nu2,norm]
new_row = [str(col) for col in new_row]
print(len(new_row))
spamwriter.writerow(new_row)
# new_line = str(test_acc) + str(np.mean(accuracy["all"]))
# new_line += "," + str(n_neurons) + "," + str(exc) + "," + str(inh)
# new_line += "," + str(nu1) + "," + str(nu2) + "," + str(norm)

# f.write(new_line)
f.close()
