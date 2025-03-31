import torch
import torch.nn.functional as F

import os
import gc
import numpy as np

import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec

from mamba_exp.attn_map_utils import ssd_attn_map, gqa_attn_map

if __name__ == "__main__":
    num_layers = 32
    seq_len = 8192
    thershold = -3

    # TODO: Modify the directories for logits loading and figs storing
    logits_dir = '/logits/dir'
    figs_dir = f'/figs/dir/{seq_len}'
    input_dir = '/input/dir'
    input_pb = torch.load(input_dir, map_location=torch.device('cpu'))
    seq_len_real = int(torch.sum(input_pb['attention_mask']).item())
    # depth = input_pb['depth']

    # Sanity Check on the input sequence
    # print(f'depth = {depth}')
    # print(input_pb['input_raw'])
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i in range(num_layers):
        print(f"---Layer {i}---")
        curr_state = torch.load(os.path.join(logits_dir, f'logit_dict_{seq_len}_{i}.pt'), map_location=torch.device('cpu'))
        if i not in [9, 18, 27]:
            dt = curr_state['dt'].detach().to(device) # (batch, seqlen, nheads)
            A = curr_state['A'].detach().to(device) # (nheads)
            B = curr_state['B'].detach().to(device) # (batch, seqlen, ngroups=1, nheads)
            C = curr_state['C'].detach().to(device) # (batch, seqlen, ngroups=1, nheads)
            dt_bias = curr_state['dt_bias'].detach().to(device) # (nheads)

            attn_map, _, dt_sp = ssd_attn_map(dt, A, B, C, dt_bias, True, (0.0, float("inf")))
            attn_map = attn_map[0].permute([1, 0, 2]).cpu()

            # Compute the recurrent states
            dA = torch.exp(torch.einsum("h, bsh -> bsh", A, dt_sp)).cpu()
            dB = torch.einsum("bshn, bsh -> bshn", B, dt_sp)
            dB_norm = torch.sqrt((dB ** 2).sum(dim=-1)).cpu()
            C_norm = torch.sqrt((C[0, :, 0, :] ** 2).sum(dim=-1)).cpu()
            dAB_ratio = dA/(dA+dB_norm)
        else:
            q = curr_state['q'].detach().to(device)
            k = curr_state['k'].detach().to(device)
            v = curr_state['v'].detach().to(device)
            mask = curr_state['mask'].detach().to(device)
            scaling = curr_state['scaling']
            attn_map = gqa_attn_map(q, k, mask, scaling)[0].cpu()

        attn_map = attn_map[..., -seq_len_real:, -seq_len_real:]

        figs_dir_layer = os.path.join(figs_dir, f'layer_{i}')
        os.makedirs(figs_dir_layer, exist_ok=True)

        for i_head in range(attn_map.shape[0]):
            # For some reason, the attention layers also need the following normalization
            attn_map_n = attn_map[i_head,:,:]/torch.max(attn_map[i_head,:,:])
            attn_map_n = torch.max(
                thershold*torch.ones(attn_map_n.shape),
                torch.log10(torch.abs(attn_map_n)).to(torch.float32)
                )
            
            if i not in [9, 18, 27]:
                dt_seq = dt[0, :, i_head].cpu().numpy()
                dt_sp_seq = dt_sp[0, :, i_head].cpu().numpy()
                dA_seq = dA[0, :, i_head].cpu().numpy()
                dB_seq = dB_norm[0, :, i_head].cpu().numpy()
                dr_seq = dAB_ratio[0, :, i_head].cpu().numpy()

                # Create a figure with three subplots (one row per plot).
                fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))
                colors = np.linspace(0, 1, len(dt_seq))

                # Plot dt_seq vs dA_seq
                axs[0].scatter(dt_seq, dA_seq, c=colors, cmap='turbo', marker='o')
                axs[0].set_title('Forget Gate Magnitude')
                axs[0].set_xlabel('dt')
                axs[0].set_ylabel('dA')
                axs[0].grid(True)

                # Plot dt_seq vs dB_seq
                axs[1].scatter(dt_seq, dB_seq, c=colors, cmap='turbo', marker='o')
                axs[1].set_title('Input Gate Magnitude')
                axs[1].set_xlabel('dt')
                axs[1].set_ylabel('dB')
                axs[1].grid(True)

                # Plot dt_seq vs dr_seq
                axs[2].scatter(dt_seq, dr_seq, c=colors, cmap='turbo', marker='o')
                axs[2].set_title('Forget Gate Ratio')
                axs[2].set_xlabel('dt')
                axs[2].set_ylabel('dA/(dA+dB)')
                axs[2].grid(True)

                plt.tight_layout()

                fig.savefig(os.path.join(figs_dir_layer, f'head_{i_head}_sequences.png'), dpi=600)
                plt.close(fig)

                # Create a figure with three subplots (one row per plot).
                fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))
                colors = np.linspace(0, 1, len(dt_seq))

                # Plot dt_seq vs dA_seq
                axs[0].plot(colors, dt_sp_seq,linestyle='-')
                axs[0].set_title('Delta post softplus at each position')
                axs[0].set_xlabel('token position')
                axs[0].set_ylabel('delta')
                axs[0].grid(True)

                # Plot dt_seq vs dB_seq
                axs[1].plot(colors, dB_seq, linestyle='-')
                axs[1].plot(colors, dA_seq, linestyle='-')
                axs[1].set_title('Input and Forget Gate Magnitude')
                axs[1].set_xlabel('token position')
                axs[1].set_ylabel('magnitude')
                axs[1].grid(True)

                # Plot dt_seq vs dr_seq
                axs[2].scatter(colors, dr_seq, linestyle='-')
                axs[2].set_title('Forget Gate Ratio')
                axs[2].set_xlabel('token position')
                axs[2].set_ylabel('dA/(dA+dB)')
                axs[2].grid(True)

                plt.tight_layout()

                fig.savefig(os.path.join(figs_dir_layer, f'head_{i_head}_position.png'), dpi=600)
                plt.close(fig)
            # else:
            #     attn_map_n = torch.clamp(attn_map[i_head], 0.0, float("inf")) 
                
            fig, ax = plt.subplots()
            # Convert to numpy array for plotting; adjust the colormap if desired.
            im = ax.imshow(attn_map_n.cpu().numpy(), cmap='turbo', vmin=-3, vmax=0)
            ax.set_box_aspect(1)  # Ensure the plot is square
            fig.colorbar(im, ax=ax)

            fig.savefig(os.path.join(figs_dir_layer, f'head_{i_head}.png'))
            plt.close(fig)

        # output the average attention map for layer-wise comparison
        attn_map_avg = torch.mean(attn_map, dim=0)

        if i not in [9, 18, 27]:
            attn_map_avg = attn_map_avg/torch.max(attn_map_avg)
            attn_map_avg = torch.max(
                thershold*torch.ones(attn_map_avg.shape),
                torch.log10(torch.abs(attn_map_avg)).to(torch.float32)
                )

        # Plot and save the average attention map
        fig, ax = plt.subplots()
        im = ax.imshow(attn_map_avg.cpu().numpy(), cmap='turbo')
        ax.set_box_aspect(1)
        fig.colorbar(im, ax=ax)
        figs_dir_avg = os.path.join(figs_dir, 'avg')
        os.makedirs(figs_dir_avg, exist_ok=True)
        fig.savefig(os.path.join(figs_dir_avg, f'layer_{i}.png'))
        plt.close(fig)
        
        gc.collect()
        torch.cuda.empty_cache()
