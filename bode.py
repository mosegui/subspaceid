import matplotlib.pyplot as plt
import scipy.signal

def bode_plot(a, b, c, d, dt):

    if b.shape[-1] != d.shape[-1]:
        raise ValueError('D and B matrices must have same number of columns')

    number_of_inputs = b.shape[-1]
    number_of_outputs = c.shape[0]

    ws = []
    mags = []
    phases = []
    labels = []

    for col in range(number_of_inputs):
        for row in range(number_of_outputs):
            c_row = c[row].reshape(1, -1)
            b_col = b[:, col].reshape(-1,1)
            d_col = d[row, col].reshape(-1,1)

            w_col, mag_col, phase_col = scipy.signal.dbode((a, b_col, c_row, d_col, dt))
            ws.append(w_col)
            mags.append(mag_col)
            phases.append(phase_col)
            labels.append(f"I{col+1} -> O{row+1}")

    fig, (ax_mag, ax_phase) = plt.subplots(nrows=2)

    for idx, w in enumerate(ws):
        ax_mag.semilogx(w, mags[idx], label=labels[idx])
        ax_phase.semilogx(w, phases[idx], label=labels[idx])
    ax_mag.legend()
    ax_phase.legend()
    plt.show()
