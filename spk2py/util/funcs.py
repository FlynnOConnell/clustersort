import numpy as np


def get_spikes(data, spike_window=80, tf=5, offset=10, max_thresh=350):
    # Calculate threshold based on data mean
    thresh = np.mean(np.abs(data)) * tf

    # Find positions wherere the threshold is crossed
    pos = np.where(data > thresh)[0]
    pos = pos[pos > spike_window]

    # Extract potential spikes and align them to the maximum
    spike_samp = []
    wave_form = np.empty([1, spike_window * 2])
    for i in pos:
        if i < data.shape[0] - (spike_window + 1):
            # Data from position where threshold is crossed to end of window
            tmp_waveform = data[i:i + spike_window * 2]

            # Check if data in window is below upper threshold (artifact rejection)
            if np.max(tmp_waveform) < max_thresh:
                # Find sample with maximum data point in window
                tmp_samp = np.argmax(tmp_waveform) + i

                # Re-center window on maximum sample and shift it by offset
                tmp_waveform = data[tmp_samp - (spike_window - offset):tmp_samp + (spike_window + offset)]

                # Append data
                try:
                    spike_samp = np.append(spike_samp, tmp_samp)
                    wave_form = np.append(wave_form, tmp_waveform.reshape(1, spike_window * 2), axis=0)
                except Exception:
                    spike_samp = [tmp_samp]
                    wave_form = tmp_waveform.reshape(1, spike_window * 2)

    # Remove duplicates
    ind = np.where(np.diff(spike_samp) > 1)[0]
    if ind.size > 0:
        spike_samp = spike_samp[ind]
        wave_form = wave_form[ind]
    return spike_samp, wave_form
