"""Run EEG."""

import pylsl
from pylsl import StreamInlet, resolve_stream
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time


import pylsl
    



def main():
    # Resolve EEG streams
    streams = pylsl.resolve_streams()
    # Iterate through streams
    print(f"Found {len(streams)} streams")
    print("---------------")
    for stream in streams:
        print("Stream Name:", stream.name())
        print("Stream Type:", stream.type())
        print("Stream ID:", stream.source_id())   # this should match your X.on Serial Number
        print("Stream Unique Identifier:", stream.uid())
        print("---------------")
        
    # print("Looking for EEG streams...")
    # streams = resolve_stream('type', 'EEG')
    # print('Resolved streams')
    # target_stream_name = "X.on-102106-0035"
    # target_stream = None
    # for stream in streams:
    #     if stream.name() == target_stream_name:
    #         target_stream = stream
    #         break
    # print('Target')
    # if target_stream:
    #     inlet = StreamInlet(target_stream)
    #     print('Made Inlet')
    #     # inlet = pylsl.stream_inlet(target_stream, 0.5)
        
        
    #     samples = []
    #     timestamps = []
    #     print('Starting')
    #     start_time = time.time()
    #     while time.time() < start_time + 1:
    #         # Get a new sample
    #         sample, timestamp = inlet.pull_sample()
    #         samples.append(sample)
    #         timestamps.append(timestamp)
        
    #     print(timestamps)
            
    #     # samples = np.array(samples)
    #     # timestamps = np.array(timestamps)
    #     # print(samples.shape)
        
    #     # fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    #     # axes[1].plot(timestamps)
    #     # axes[0].plot(samples)
    #     # plt.show()
        
        
    #     # import pdb; pdb.set_trace()
    #     # a = 3
    # else:
    #     print(f"No stream found with name {target_stream_name}. Please check the stream name and try again.")


if __name__ == '__main__':
    main()