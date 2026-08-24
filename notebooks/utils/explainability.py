import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

'''
def saliency_map(model, ecg, target_class="MI"):
    """
    Computes a saliency map for all 12 ECG leads.

    Parameters
    ----------
    model : tf.keras.Model
        Trained binary classification model.

    ecg : np.ndarray
        ECG signal with shape (timesteps, leads).

    target_class : str
        Class being explained: "MI" or "NORM".

    Returns
    -------
    probability_mi : float
        Probability assigned to MI.

    saliency : np.ndarray
        Saliency map with shape (timesteps, leads).
    """

    x = tf.convert_to_tensor(
        ecg[None, ...],
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:
        tape.watch(x)

        prediction = model(x, training=False)
        probability_mi = prediction[:, 0]

        if target_class == "MI":
            target_score = probability_mi

        elif target_class == "NORM":
            target_score = 1.0 - probability_mi

        else:
            raise ValueError(
                "target_class must be 'MI' or 'NORM'"
            )

        

    gradients = tape.gradient(target_score, x)

    saliency = np.abs(
        gradients[0].numpy()
    )

    return float(probability_mi.numpy()[0]), saliency

def plot_ecg_saliency(
    model,
    ecg,
    true_class=None,
    target_class="MI"
):
    """
    Plots all ECG leads with their saliency values.
    """

    probability_mi, saliency = saliency_map(
        model,
        ecg,
        target_class=target_class
    )

    predicted_class = (
        "MI"
        if probability_mi >= 0.5
        else "NORM"
    )

    lead_names = [
        "I", "II", "III",
        "aVR", "aVL", "aVF",
        "V1", "V2", "V3",
        "V4", "V5", "V6"
    ]

    # Normaliza a saliência pra [0, 1] pra poder virar cor de fundo
    saliency_norm = (
        saliency / saliency.max()
        if saliency.max() > 0
        else saliency
    )

    # Suaviza pra ficar contínuo, igual no GRAD-CAM
    saliency_smooth = gaussian_filter1d(saliency_norm, sigma=2.0, axis=0)

    cmap = plt.cm.Reds  # branco (0) -> vermelho (1)

    fig, axes = plt.subplots(
        12,
        1,
        figsize=(15, 24),
        sharex=True
    )

    time = np.arange(ecg.shape[0])

    for lead in range(12):

        ax = axes[lead]

        # ECG waveform
        ax.plot(
            time,
            ecg[:, lead],
            color="black",
            linewidth=0.8,
            zorder=2
        )

        # Saliência como fundo contínuo, igual no GRAD-CAM
        for t in range(len(time) - 1):
            color = cmap(saliency_smooth[t, lead])
            ax.axvspan(
                time[t], time[t + 1],
                facecolor=color, alpha=0.5, zorder=1
            )

        ax.set_ylabel(
            lead_names[lead],
            rotation=0,
            labelpad=25
        )

        ax.grid(alpha=0.2)

    axes[-1].set_xlabel(
        "Tempo (amostras)"
    )

    fig.suptitle(
        f"ECG + Saliency Map\n"
        f"P(MI) = {probability_mi:.3f} | "
        f"Predição = {predicted_class}"
        + (
            f" | Classe real = {true_class}"
            if true_class is not None
            else ""
        ),
        fontsize=14
    )

    plt.tight_layout()
    plt.show()
'''

def saliency_map(model, ecg, target_class="MI"):
    """
    Computes a saliency map for all 12 ECG leads.

    Parameters
    ----------
    model : tf.keras.Model
        Trained binary classification model.

    ecg : np.ndarray
        ECG signal with shape (timesteps, leads).

    target_class : str
        Class being explained: "MI" or "NORM".

    Returns
    -------
    probability_mi : float
        Probability assigned to MI.

    saliency : np.ndarray
        Saliency map with shape (timesteps, leads).
    """

    x = tf.convert_to_tensor(
        ecg[None, ...],
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:
        tape.watch(x)

        prediction = model(x, training=False)
        probability_mi = prediction[:, 0]

        if target_class == "MI":
            target_score = probability_mi

        elif target_class == "NORM":
            target_score = 1.0 - probability_mi

        else:
            raise ValueError(
                "target_class must be 'MI' or 'NORM'"
            )

    gradients = tape.gradient(target_score, x)

    saliency = np.abs(
        gradients[0].numpy()
    )

    return float(probability_mi.numpy()[0]), saliency

def plot_ecg_saliency(
    model,
    ecg,
    true_class=None,
    target_class="MI",
    dir_path = None
):
    """
    Plots all ECG leads with their saliency values.
    """

    probability_mi, saliency = saliency_map(
        model,
        ecg,
        target_class=target_class
    )

    predicted_class = (
        "MI"
        if probability_mi >= 0.5
        else "NORM"
    )

    lead_names = [
        "I", "II", "III",
        "aVR", "aVL", "aVF",
        "V1", "V2", "V3",
        "V4", "V5", "V6"
    ]

    fig, axes = plt.subplots(
        12,
        1,
        figsize=(15, 24),
        sharex=True
    )

    time = np.arange(ecg.shape[0])

    for lead in range(12):

        ax = axes[lead]

        # ECG waveform
        ax.plot(
            time,
            ecg[:, lead],
            linewidth=0.8
        )

        # Saliency represented by point intensity
        ax.scatter(
            time,
            ecg[:, lead],
            c=saliency[:, lead],
            cmap="hot",
            s=5
        )

        ax.set_ylabel(
            lead_names[lead],
            rotation=0,
            labelpad=25
        )

        ax.grid(alpha=0.2)

    axes[-1].set_xlabel(
        "Tempo (amostras)"
    )

    fig.suptitle(
        f"ECG + Saliency Map\n"
        f"P(MI) = {probability_mi:.3f} | "
        f"Predição = {predicted_class}"
        + (
            f" | Classe real = {true_class}"
            if true_class is not None
            else ""
        ),
        fontsize=14
    )

    plt.tight_layout()
    plt.show()

    if not dir_path == None:
        filename = f'{predicted_class}_probMI_{probability_mi}'
        fig.savefig(f'{dir_path}\\{filename}.jpg')





def get_gradcam_improved(model, input_sample, class_idx=0, 
                        smoothing='gaussian', smooth_sigma=3.0,
                        edge_handling='trim', trim_percent=0.05,
                        normalize_method='global'):
    """
    Improved GRAD-CAM implementation with better edge handling.
    
    Args:
        model: Trained Keras model
        input_sample: Input sample of shape (1, timesteps, features)
        class_idx: Class index to explain (0=NORM, 1=MI)
        smoothing: 'gaussian', 'moving_avg', or None
        smooth_sigma: Sigma for gaussian smoothing (or window size for moving avg)
        edge_handling: 'trim' (remove edges), 'taper' (fade edges), or 'none'
        trim_percent: Percentage of signal to trim from each edge
        normalize_method: 'global' (all leads together) or 'per_lead'
    
    Returns:
        heatmap_2d: Heatmap of shape (timesteps, 12)
        edge_mask: Boolean mask indicating valid (non-edge) regions
    """
    # Compute gradients
    with tf.GradientTape() as tape:
        inputs = tf.convert_to_tensor(input_sample, dtype=tf.float32)
        tape.watch(inputs)
        predictions = model(inputs, training=False)
        
        if class_idx == 1:
            class_score = predictions[:, 0]
        else:
            class_score = 1 - predictions[:, 0]
    
    grads = tape.gradient(class_score, inputs)
    
    if grads is None:
        timesteps = input_sample.shape[1]
        n_leads = input_sample.shape[2]
        return np.ones((timesteps, n_leads)) / (timesteps * n_leads), np.ones(timesteps, dtype=bool)
    
    grads_np = grads.numpy()[0]
    inputs_np = input_sample[0]
    
    # Compute importance: |gradient| * |input|
    importance = np.abs(grads_np) * np.abs(inputs_np)
    
    timesteps = importance.shape[0]
    n_leads = importance.shape[1]
    
    # Create edge mask
    trim_samples = int(timesteps * trim_percent)
    edge_mask = np.ones(timesteps, dtype=bool)
    
    if edge_handling == 'trim' and trim_samples > 0:
        edge_mask[:trim_samples] = False
        edge_mask[-trim_samples:] = False
    elif edge_handling == 'taper' and trim_samples > 0:
        # Create taper weights (0 at edges, 1 in middle)
        taper = np.ones(timesteps)
        taper[:trim_samples] = np.linspace(0, 1, trim_samples)
        taper[-trim_samples:] = np.linspace(1, 0, trim_samples)
        importance = importance * taper[:, np.newaxis]
    
    # Apply smoothing
    heatmap_2d = np.zeros_like(importance)
    
    if smoothing == 'gaussian':
        for lead_idx in range(n_leads):
            heatmap_2d[:, lead_idx] = gaussian_filter1d(
                importance[:, lead_idx], sigma=smooth_sigma
            )
    elif smoothing == 'moving_avg':
        window_size = int(smooth_sigma)
        kernel = np.ones(window_size) / window_size
        for lead_idx in range(n_leads):
            # Use 'same' but handle edges differently
            smoothed = np.convolve(importance[:, lead_idx], kernel, mode='same')
            heatmap_2d[:, lead_idx] = smoothed
    else:
        heatmap_2d = importance.copy()
    
    # Normalize
    if normalize_method == 'global':
        # Normalize globally - preserves relative importance between leads
        valid_data = heatmap_2d[edge_mask, :]
        if valid_data.max() > 0:
            heatmap_2d = heatmap_2d / valid_data.max()
    else:
        # Normalize per lead
        for lead_idx in range(n_leads):
            valid_data = heatmap_2d[edge_mask, lead_idx]
            if valid_data.max() > 0:
                heatmap_2d[:, lead_idx] = heatmap_2d[:, lead_idx] / valid_data.max()
    
    return heatmap_2d, edge_mask



def plot_ecg_with_gradcam(sample, heatmap_2d, true_label, pred_prob, sample_idx, 
                          lead_names=None, smooth_sigma=2.0, dir_path=None):
    """
    Plot all 12 ECG leads with GRAD-CAM heatmap overlay.
    
    Args:
        sample: ECG sample of shape (timesteps, 12)
        heatmap_2d: GRAD-CAM heatmap of shape (timesteps, 12)
        true_label: Ground truth label (0=NORM, 1=MI)
        pred_prob: Predicted probability for MI class
        sample_idx: Sample index for title
        lead_names: Names of the 12 leads
        smooth_sigma: Gaussian smoothing parameter for heatmap
    """
    
    n_leads = sample.shape[1]
    timesteps = sample.shape[0]
    time_axis = np.arange(timesteps) / 100  # Convert to seconds (100 Hz sampling)
    
    # Create color map: blue (NORM) to white to red (MI)
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # Blue -> White -> Red
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('norm_mi', colors, N=n_bins)
    
    # Smooth the heatmap for better visualization
    heatmap_smooth = gaussian_filter1d(heatmap_2d, sigma=smooth_sigma, axis=0)
    
    # Create figure with 12 subplots (one per lead)
    fig, axes = plt.subplots(12, 1, figsize=(18, 24), sharex=True)
    
    for i in range(n_leads):
        ax = axes[i]
        
        # Plot the ECG signal
        ax.plot(time_axis, sample[:, i], color='black', linewidth=1.0, zorder=2)
        
        # Create a gradient background based on GRAD-CAM importance
        for t in range(timesteps - 1):
            importance = heatmap_smooth[t, i]
            color = cmap(importance)
            ax.axvspan(time_axis[t], time_axis[t+1], 
                      facecolor=color, alpha=0.5, zorder=1)
        
        # Formatting
        ax.set_ylabel(f'{lead_names[i]}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_xlim(0, 10)
        
        # Hide x-axis tick labels for all but the bottom subplot
        ax.tick_params(axis='x', labelbottom=False)
    
    # Set x-label and ticks for the bottom subplot only
    axes[-1].tick_params(axis='x', labelbottom=True)
    axes[-1].set_xlabel('Tempo (segundos)', fontsize=14, fontweight='bold')
    axes[-1].set_xticks(np.arange(0, 11, 1))
    
    # Add overall title with prediction information
    true_class = 'MI' if true_label == 1 else 'NORM'
    pred_class = 'MI' if pred_prob > 0.5 else 'NORM'
    
    title_text = (f'GRAD-CAM: Explicação Visual para Classificação ECG (Amostra #{sample_idx})\n'
                  f'Classe Real: {true_class} | Predição: {pred_class} '
                  f'(P(MI) = {pred_prob:.4f})\n'
                  f'Vermelho = Importante para MI | Azul = Importante para NORM')
    
    fig.suptitle(title_text, fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.show()

    if not dir_path == None:
        filename = f'{pred_class}_probMI_{pred_prob}'
        fig.savefig(f'{dir_path}\\{filename}.jpg')



def plot_lead_importance_ranking(heatmap_2d, sample_idx=None, lead_names=None):
    """
    Plot a bar chart showing the importance ranking of each lead.
    
    Args:
        heatmap_2d: GRAD-CAM heatmap of shape (timesteps, 12)
        sample_idx: Index of the sample being analyzed
        lead_names: Names of the 12 leads
    """
    
    # Compute average importance for each lead
    lead_importance = np.mean(heatmap_2d, axis=0)
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'Lead': lead_names,
        'Importance': lead_importance
    })
    df = df.sort_values('Importance', ascending=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use horizontal bar plot with better colors
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df)))
    bars = ax.barh(df['Lead'], df['Importance'], color=colors, 
                   edgecolor='black', linewidth=0.5, alpha=0.85)
    
    # Add value labels on bars
    for bar, score in zip(bars, df['Importance']):
        width = bar.get_width()
        ax.text(width + 0.002, bar.get_y() + bar.get_height()/2, 
                f'{score:.4f}', va='center', fontsize=9, alpha=0.8)
    
    ax.set_xlabel('Score de Importância', fontsize=12)
    ax.set_ylabel('Derivação', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    
    # Adjust x-axis to accommodate labels
    max_score = df['Importance'].max()
    ax.set_xlim(0, max_score * 1.15)
    
    # Invert y-axis so highest values are at the top
    ax.invert_yaxis()
    
    title = 'Importância Média das Derivações (GRAD-CAM)'
    if sample_idx is not None:
        title += f' - Amostra #{sample_idx}'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Score de Importância', fontsize=12)
    plt.ylabel('Derivação', fontsize=12)
    plt.tight_layout()
    plt.show()

    return df



def plot_temporal_importance(heatmap_2d, sample_idx=None):
    """
    Plot the average temporal importance across all leads.
    
    Args:
        heatmap_2d: GRAD-CAM heatmap of shape (timesteps, 12)
        sample_idx: Index of the sample being analyzed
    """
    # Average importance across all leads for each timestep
    temporal_importance = np.mean(heatmap_2d, axis=1)
    time_axis = np.arange(len(temporal_importance)) / 100  # Convert to seconds
    
    plt.figure(figsize=(14, 5))
    plt.plot(time_axis, temporal_importance, linewidth=2, color='darkblue')
    plt.fill_between(time_axis, 0, temporal_importance, alpha=0.3, color='skyblue')
    
    title = 'Importância Temporal Média (GRAD-CAM)'
    if sample_idx is not None:
        title += f' - Amostra #{sample_idx}'
    plt.title(title, fontsize=14, fontweight='bold')

    plt.xlabel('Tempo (segundos)', fontsize=12)
    plt.ylabel('Score de Importância', fontsize=12)    
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Set x-axis ticks from 0 to 10 seconds
    plt.xticks(np.arange(0, 11, 1), fontsize=10)
    plt.xlim(0, 10)
    
    plt.tight_layout()
    plt.show()