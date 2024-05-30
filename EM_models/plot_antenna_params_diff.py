# Copyright (C) 2015-2023: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax.  If not, see <http://www.gnu.org/licenses/>.

from plot_antenna_params import calculate_antenna_params, mpl_plot
import argparse
import os
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gprMax.exceptions import CmdInputError


def mpl_plot_diff(filename1, filename2, time, freqs, Vinc, Vincp, Iinc, Iincp, Vref, Vrefp, Iref, Irefp, Vtotal, Vtotalp, Itotal, Itotalp, s11, zin, zrec, yin, Vrec, Vrecp, Irec, Irecp, s21=None):
    """Plots antenna parameters - incident, reflected and total volatges and currents; s11, (s21) and input impedance.

    Args:
        filename (string): Filename (including path) of output file.
        time (array): Simulation time.
        freq (array): Frequencies for FFTs.
        Vinc, Vincp, Iinc, Iincp (array): Time and frequency domain representations of incident voltage and current.
        Vref, Vrefp, Iref, Irefp (array): Time and frequency domain representations of reflected voltage and current.
        Vtotal, Vtotalp, Itotal, Itotalp (array): Time and frequency domain representations of total voltage and current.
        s11, s21 (array): s11 and, optionally, s21 parameters.
        zin, yin (array): Input impedance and input admittance parameters.

    Returns:
        plt (object): matplotlib plot object.
    """

    dirname1 = os.path.dirname(os.path.relpath(args.outputfile1)).split('/')[-1]
    dirname2 = os.path.dirname(os.path.relpath(args.outputfile2)).split('/')[-1]

    # Set plotting range
    pltrangemin = 1
    # To a certain drop from maximum power
    pltrangemax = np.where((np.amax(Vrecp[1::]) - Vrecp[1::]) > 60)[0][0] + 1
    # To a maximum frequency
    # pltrangemax = np.where(freqs > 6e9)[0][0]
    pltrange = np.s_[pltrangemin:pltrangemax]

    # Print some useful values from s11, and input impedance
    s11minfreq = np.where(s11[pltrange] == np.amin(s11[pltrange]))[0][0]
    print('s11 minimum: {:g} dB at {:g} Hz'.format(np.amin(s11[pltrange]), freqs[s11minfreq + pltrangemin]))
    print('At {:g} Hz...'.format(freqs[s11minfreq + pltrangemin]))
    print('Input impedance: {:.1f}{:+.1f}j Ohms'.format(np.abs(zin[s11minfreq + pltrangemin]), zin[s11minfreq + pltrangemin].imag))
    # print('Input admittance (mag): {:g} S'.format(np.abs(yin[s11minfreq + pltrangemin])))
    # print('Input admittance (phase): {:.1f} deg'.format(np.angle(yin[s11minfreq + pltrangemin], deg=True)))

    # Figure 1
    # Plot incident voltage
    fig1, ax = plt.subplots(num='Transmitter transmission line parameters', figsize=(30, 8), facecolor='w', edgecolor='w')
    ax.axis('off')
    gs1 = gridspec.GridSpec(2, 2, hspace=0.7)

    # Plot received  voltage
    ax = plt.subplot(gs1[0, 0])
    ax.plot(time, Vrec, 'r', lw=2, label='Vref')
    ax.set_title('Received voltage difference: ' + dirname1 + ' - ' + dirname2)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Voltage [V]')
    ax.set_xlim([0, np.amax(time)])
    ax.grid(which='both', axis='both', linestyle='-.')

    ## Plot frequency spectra of received voltage
    ax = plt.subplot(gs1[0, 1])
    markerline, stemlines, baseline = ax.stem(freqs[pltrange], Vrecp[pltrange], '-.')
    plt.setp(baseline, 'linewidth', 0)
    plt.setp(stemlines, 'color', 'r')
    plt.setp(markerline, 'markerfacecolor', 'r', 'markeredgecolor', 'r')
    ax.plot(freqs[pltrange], Vrecp[pltrange], 'r', lw=2)
    ax.set_title('Received voltage difference: ' + dirname1 + ' - ' + dirname2)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dB]')
    ax.grid(which='both', axis='both', linestyle='-.')


    # Plot received current
    ax = plt.subplot(gs1[1, 0])
    ax.plot(time, Irec, 'b', lw=2, label='Irec')
    ax.set_title('Received current difference: ' + dirname1 + ' - ' + dirname2)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Current [A]')
    ax.set_xlim([0, np.amax(time)])
    ax.grid(which='both', axis='both', linestyle='-.')

    ## Plot frequency spectra of received current
    ax = plt.subplot(gs1[1, 1])
    markerline, stemlines, baseline = ax.stem(freqs[pltrange], Irecp[pltrange], '-.')
    plt.setp(baseline, 'linewidth', 0)
    plt.setp(stemlines, 'color', 'b')
    plt.setp(markerline, 'markerfacecolor', 'b', 'markeredgecolor', 'b')
    ax.plot(freqs[pltrange], Irecp[pltrange], 'b', lw=2)
    ax.set_title('Received current difference: ' + dirname1 + ' - ' + dirname2)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dB]')
    ax.set_xlim([0,1e9])
    ax.grid(which='both', axis='both', linestyle='-.')


    # Save a PDF/PNG of the figure
    # fig1.savefig(os.path.splitext(os.path.abspath(filename))[0] + '_tl_params.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
    # fig2.savefig(os.path.splitext(os.path.abspath(filename))[0] + '_ant_params.png', dpi=150, format='png', bbox_inches='tight', pad_inches=0.1)
    # dirname = os.path.dirname(os.path.relpath(filename)).split('/')[-1]
    fig1.savefig('diff_' + dirname1 + '-' + dirname2 + '_tl_params.pdf', dpi=None, format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig1)

    return plt


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plots antenna parameters (s11, s21 parameters and input impedance) from an output file containing a transmission line source.', usage='cd gprMax; python -m tools.plot_antenna_params outputfile')
    parser.add_argument('outputfile1', help='name of output file including path')
    parser.add_argument('outputfile2', help='name of output file including path')
    parser.add_argument('--tltx-num', default=1, type=int, help='transmitter antenna - transmission line number')
    parser.add_argument('--tlrx-num', type=int, help='receiver antenna - transmission line number')
    parser.add_argument('--rx-num', type=int, help='receiver antenna - output number')
    parser.add_argument('--rx-component', type=str, help='receiver antenna - output electric field component', choices=['Ex', 'Ey', 'Ez'])
    args = parser.parse_args()

    antennaparams1 = calculate_antenna_params(args.outputfile1, args.tltx_num, args.tlrx_num, args.rx_num, args.rx_component)
    antennaparams2 = calculate_antenna_params(args.outputfile2, args.tltx_num, args.tlrx_num, args.rx_num, args.rx_component)
    antennaparams = {k:(v-antennaparams2[k]) for k,v in antennaparams1.items()}
    antennaparams['time'] = antennaparams1['time']
    antennaparams['freqs'] = antennaparams1['freqs']
    # print(antennaparams1)
    # print(antennaparams)
    mpl_plot(args.outputfile1, **antennaparams1)
    mpl_plot(args.outputfile2, **antennaparams2)
    mpl_plot_diff(args.outputfile1, args.outputfile2, **antennaparams)
