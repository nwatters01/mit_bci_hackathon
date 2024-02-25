"""LSL Viewer.

An easy to use lsl signal viewer. Not suitable for use while running
 BciPy experiments.

Adapted from: https://github.com/NeuroTechX/bci-workshop
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, lfilter_zi, firwin
from time import sleep
from pylsl import StreamInlet, resolve_stream
import seaborn as sns
import feature_extractors

import logging

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)

sns.set(style="whitegrid")


class LSLViewer():

    def __init__(self,
                 stream,
                 feature_extractor,
                 window=5.,
                 scale=100.,
                 buf=1,
                 subsample=3,
                 filter_data=True,
                 dejitter=True):
        """Init"""
        self._feature_extractor = feature_extractor
        self._n_feat = self._feature_extractor.n_features
        self.stream = stream
        self.window = window
        self.scale = scale
        self.dejitter = dejitter
        self.inlet = StreamInlet(stream, max_buflen=buf, max_chunklen=buf)
        self.filt = filter_data
        self._subsample = subsample
        
        
        self._fig, axes = plt.subplots(2, 1, figsize=(6, 8))
        self._ax_raw, self._ax_feat = axes
        self._fig.canvas.mpl_connect('key_press_event', self.OnKeypress)
        self._fig.canvas.mpl_connect('key_release_event', self.OnKeyrelease)

        info = self.inlet.info()
        description = info.desc()

        self.sfreq = info.nominal_srate()
        self.n_samples = int(self.sfreq * self.window)
        self.n_chan = info.channel_count()

        ch = description.child('channels').first_child()
        ch_names = [ch.child_value('label')]

        for i in range(self.n_chan):
            ch = ch.next_sibling()
            ch_names.append(ch.child_value('label'))

        self.ch_names = ch_names

        sns.despine(left=True)

        self.data = np.zeros((self.n_samples, self.n_chan))
        self.data_f = np.zeros((self.n_samples, self.n_chan))
        self._features = np.zeros((self.n_samples, self._n_feat))
        self.times = np.arange(-self.window, 0, 1./self.sfreq)
        impedances = np.std(self.data, axis=0)
        
        # Data_f lines
        lines = []
        for ii in range(self.n_chan):
            line, = self._ax_raw.plot(self.times[::self._subsample],
                              self.data[::self._subsample, ii] - ii, lw=1)
            lines.append(line)
        self.lines = lines
        self._ax_raw.set_ylim(-self.n_chan - 0.5, 0.5)
        self._ax_raw.set_xlabel('Time (s)')
        self._ax_raw.xaxis.grid(False)
        self._ax_raw.set_yticks(np.arange(0, -self.n_chan, -1))
        ticks_labels = ['%s - %.1f' % (ch_names[ii], impedances[ii])
                        for ii in range(self.n_chan)]
        self._ax_raw.set_yticklabels(ticks_labels)
        
        # Feature lines
        lines_feat = []
        for ii in range(self._n_feat):
            line, = self._ax_feat.plot(self.times[::self._subsample],
                              self.data[::self._subsample, ii] - ii, lw=1)
            lines_feat.append(line)
        self.lines_feat = lines_feat
        self._ax_feat.set_ylim(-self._n_feat - 0.5, 0.5)
        self._ax_feat.set_xlabel('Time (s)')
        self._ax_feat.xaxis.grid(False)
        self._ax_feat.set_yticks(np.arange(0, -self._n_feat, -1))

        # Raw data filter
        self.display_every = int(0.2 / (12/self.sfreq))
        self.bf = firwin(32, np.array([1, 40])/(self.sfreq/2.), width=0.05,
                         pass_zero=False)
        self.af = [1.0]
        zi = lfilter_zi(self.bf, self.af)
        self.filt_state = np.tile(zi, (self.n_chan, 1)).transpose()
        
    def update_plot(self):
        k = 0
        while self.started:
            samples, timestamps = self.inlet.pull_chunk(timeout=1.0,
                                                        max_samples=12)
            if timestamps:
                if self.dejitter:
                    timestamps = np.float64(np.arange(len(timestamps)))
                    timestamps /= self.sfreq
                    timestamps += self.times[-1] + 1./self.sfreq
                self.times = np.concatenate([self.times, timestamps])
                self.n_samples = int(self.sfreq * self.window)
                self.times = self.times[-self.n_samples:]
                
                # Add new data
                self.data = np.vstack([self.data, samples])
                self.data = self.data[-self.n_samples:]
                
                # Add new data_f
                filt_samples, self.filt_state = lfilter(
                    self.bf, self.af,
                    samples,
                    axis=0, zi=self.filt_state)
                self.data_f = np.vstack([self.data_f, filt_samples])
                self.data_f = self.data_f[-self.n_samples:]
                
                # Add new features
                new_features = self._feature_extractor(filt_samples)
                self._features = np.vstack([self._features, new_features])
                self._features = self._features[-self.n_samples:]
                
                k += 1
                if k == self.display_every:
                    plot_data = np.copy(self.data_f[::self._subsample])
                    plot_data -= np.nanmean(plot_data, axis=0, keepdims=True)
                    plot_data /= 3 * np.nanstd(plot_data, axis=0, keepdims=True)
                    for ii in range(self.n_chan):
                        self.lines[ii].set_xdata(
                            self.times[::self._subsample] - self.times[-1])
                        self.lines[ii].set_ydata(plot_data[:, ii] - ii)
                    self._ax_raw.set_xlim(-self.window, 0)
                    
                    plot_data = np.copy(self._features[::self._subsample])
                    plot_data -= np.nanmean(plot_data, axis=0, keepdims=True)
                    plot_data /= 3 * np.nanstd(plot_data, axis=0, keepdims=True)
                    for ii in range(self._n_feat):
                        self.lines_feat[ii].set_xdata(
                            self.times[::self._subsample] - self.times[-1])
                        self.lines_feat[ii].set_ydata(plot_data[:, ii] - ii)
                    self._ax_feat.set_xlim(-self.window, 0)
                    
                    self._fig.canvas.draw()
                    k = 0
                    plt.pause(0.01)
                    sleep(0.01)
            else:
                sleep(0.01)
                
    def OnKeypress(self, event):
        self._feature_extractor.key_press(event.key)
    
    def OnKeyrelease(self, event):
        self._feature_extractor.key_release()

    def start(self):
        self.started = True
        self.update_plot()

    def stop(self):
        self.started = False


if __name__ == '__main__':
    
    # feature_extractor = feature_extractors.DoNothing()
    # feature_extractor = feature_extractors.Gabel()
    feature_extractor = feature_extractors.DataSaver(filename='mahdi_vh')
    feature_extractor = feature_extractors.DataSaver(filename='mahdi_vh')

    logging.debug("looking for an EEG stream...")
    for _ in range(100):
        streams = resolve_stream('type', 'EEG')
        target_stream_name = "X.on-102106-0035"
        target_stream = None
        for stream in streams:
            if stream.name() == target_stream_name:
                target_stream = stream
                break
            
        if target_stream:
            logging.debug("Start aquiring data")

            lslv = LSLViewer(target_stream, feature_extractor=feature_extractor)
            lslv.start()

            plt.show()
            lslv.stop()
    
    print(f'Cannot find stream {target_stream_name}')
    