"""LSL API."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, lfilter_zi, firwin
from time import sleep
from pylsl import StreamInlet, resolve_stream
import seaborn as sns

import csv
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)

sns.set(style="whitegrid")


class Stddev():
    
    def __init__(self,
                 write_file='./data/stddevs_v0.csv',
                 timesteps_per_chunk=200,
                 history_chunks=100,
                 n_channels=7):
        self._timesteps_per_chunk = timesteps_per_chunk
        self._n_channels = n_channels
        self._recent_stddevs = []
        self._writer = csv.writer(open(write_file, 'w'))
        self._history_chunks = history_chunks
    
    def __call__(self, data):
        """Process batch of data of shape [n_timesteps, n_channels]."""
        
        # Trim unused channels
        data = data[:, :self._n_channels]
        
        # Zero-mean the data
        data -= np.nanmean(data, axis=0, keepdims=True)
        
        # Append stddevs to self._recent_stddevs
        current_stddevs = np.nanstd(data, axis=0)
        self._recent_stddevs.append(current_stddevs)
        if len(self._recent_stddevs) > self._history_chunks:
            self._recent_stddevs.pop(0)
            
        # write stddevs to csv
        to_write = [str(x) for x in current_stddevs]
        self._writer.writerow(to_write)
        
        # Compute mean stddev per channel
        mean_stddevs = np.mean(self._recent_stddevs, axis=0)
        
        # Features are normalized stddevs
        features = current_stddevs / mean_stddevs
        
        return features
    
    @property
    def n_features(self):
        return self._n_channels
    
    @property
    def timesteps_per_chunk(self):
        return self._timesteps_per_chunk
    

class LSLAPI():

    def __init__(self,
                 stream,
                 feature_extractor,
                 window=5.,
                 buf=1,
                 subsample_plot=3,
                 plot=True):
        """Init"""
        self._feature_extractor = feature_extractor
        self._n_feat = self._feature_extractor.n_features
        self._window = window
        self._inlet = StreamInlet(stream, max_buflen=buf, max_chunklen=buf)
        self._subsample_plot = subsample_plot
        self._plot = plot
        
        # General metadata about inlet stream
        info = self._inlet.info()
        self._sfreq = info.nominal_srate()
        self._n_samples = int(self._sfreq * self._window)
        self._n_chan = info.channel_count()

        # Initialize variable for data, features, and times
        self._times = np.arange(-self._window, 0, 1. / self._sfreq)
        self._data = np.zeros((self._n_samples, self._n_chan))
        self._features = np.zeros((self._n_samples, self._n_feat))
        
        # Raw data filter
        self._display_every = int(0.2 / (12 / self._sfreq))
        self._bf = firwin(
            32,
            np.array([1, 40]) / (self._sfreq / 2.),
            width=0.05,
            pass_zero=False,
        )
        self._af = [1.0]
        zi = lfilter_zi(self._bf, self._af)
        self._filt_state = np.tile(zi, (self._n_chan, 1)).transpose()
        
    def __call__(self):
        
        while self._inlet.samples > 24:
            samples, timestamps = self._inlet.pull_chunk(
                timeout=0.01, max_samples=24)
        # samples, timestamps = self._inlet.pull_chunk(
        #     timeout=0.01, max_samples=24)
        
        if timestamps:
            # Dejitter and append times
            num_new_samples = len(timestamps)
            timestamps = np.float64(np.arange(num_new_samples)) / self._sfreq
            timestamps += self._times[-1] + 1. / self._sfreq
            self._times = np.concatenate([self._times, timestamps])
            self._n_samples = int(self._sfreq * self._window)
            self._times = self._times[-self._n_samples:]
            
            # Add new data
            filt_samples, self._filt_state = lfilter(
                self._bf, self._af, samples, axis=0, zi=self._filt_state)
            self._data = np.vstack([self._data, filt_samples])
            self._data = self._data[-self._n_samples:]
        
        # Return new features
        new_features = self._feature_extractor(
            self._data[-self._feature_extractor.timesteps_per_chunk:])
        
        return new_features, num_new_samples
        
    def update_plot(self, update):
        new_features, num_new_samples = self.__call__()
        new_features_vec = (
            new_features[None] * np.ones((num_new_samples, 1)))
        self._features = np.vstack([self._features, new_features_vec])
        self._features = self._features[-self._n_samples:]
        
        # Plot if necessary
        if self._plot and update:
            plot_data = np.copy(self._data[::self._subsample_plot])
            plot_data -= np.nanmean(plot_data, axis=0, keepdims=True)
            plot_data /= 3 * np.nanstd(plot_data, axis=0, keepdims=True)
            for chan in range(self._n_chan):
                self.lines_data[chan].set_xdata(
                    self._times[::self._subsample_plot] - self._times[-1])
                self.lines_data[chan].set_ydata(plot_data[:, chan] - chan)
            self._ax_raw.set_xlim(-self._window, 0)
            
            plot_data = np.copy(self._features[::self._subsample_plot])
            plot_data -= np.nanmean(plot_data, axis=0, keepdims=True)
            plot_data /= 3 * np.nanstd(plot_data, axis=0, keepdims=True)
            for chan in range(self._n_feat):
                self.lines_feat[chan].set_xdata(
                    self._times[::self._subsample_plot] - self._times[-1])
                self.lines_feat[chan].set_ydata(plot_data[:, chan] - chan)
            self._ax_feat.set_xlim(-self._window, 0)
            
            self._fig.canvas.draw()
            plt.pause(0.01)
            sleep(0.01)
                
    def run_plot_loop(self):
        sns.despine(left=True)
        self._fig, axes = plt.subplots(2, 1, figsize=(6, 8))
        self._ax_raw, self._ax_feat = axes
        times_plot = self._times[::self._subsample_plot]
    
        # Data lines
        lines_data = []
        for channel in range(self._n_chan):
            line, = self._ax_raw.plot(
                times_plot, np.zeros_like(times_plot) - channel, lw=1)
            lines_data.append(line)
        self.lines_data = lines_data
        self._ax_raw.set_ylim(-self._n_chan - 0.5, 0.5)
        self._ax_raw.set_xlabel('Time (s)')
        self._ax_raw.xaxis.grid(False)
        self._ax_raw.set_yticks(np.arange(0, -self._n_chan, -1))
    
        # Feature lines
        lines_feat = []
        for channel in range(self._n_feat):
            line, = self._ax_feat.plot(
                times_plot, np.zeros_like(times_plot) - channel, lw=1)
            lines_feat.append(line)
        self.lines_feat = lines_feat
        self._ax_feat.set_ylim(-self._n_feat - 0.5, 0.5)
        self._ax_feat.set_xlabel('Time (s)')
        self._ax_feat.xaxis.grid(False)
        self._ax_feat.set_yticks(np.arange(0, -self._n_feat, -1))
        
        update_index = 0
        while True:
            update = update_index % self._display_every == 0
            self.update_plot(update=update)
            update += 1
            
    @property
    def n_features(self):
        return self._feature_extractor.n_features
            
            
def get_lsl_api():
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
            feature_extractor = Stddev(write_file='./data/stddevs_v0.csv')
            lsl_api = LSLAPI(target_stream, feature_extractor=feature_extractor)
            return lsl_api
    
    print(f'Cannot find stream {target_stream_name}')
    return None


class NoiseAPI():
    
    def __init__(self, n_features=7):
        self._n_features = n_features
        
    def __call__(self):
        return np.random.uniform(-1, 1, size=self._n_features)
        
    @property
    def n_features(self):
        return self._n_features


def get_noise_api():
    return NoiseAPI()


if __name__ == '__main__':
    lsl_api = get_lsl_api()
    if lsl_api is None:
        exit()
    lsl_api.run_plot_loop()
    