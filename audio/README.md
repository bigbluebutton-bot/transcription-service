# How to get a valid ogg opus file

## Install ffmpeg
```bash
sudo apt update && sudo apt install ffmpeg -y
```

## Convert to a valid audio file
```bash
$ ffmpeg -i audio.ogg -c:a libopus -frame_duration 20 -page_duration 20000 -vn out.ogg
```

## Read the header of the file
### Make sure that the file is a valid ogg file
```bash
$ ffmpeg -i out.ogg -f ffmetadata - 2>&1 | grep -A 1 "Input #0"
Input #0, ogg, from 'audio.ogg':
  Duration: 00:11:54.16, start: 0.000000, bitrate: 133 kb/s
```
If it does not show `Input #0, ogg` then the file is not a valid ogg file

### Make sure inside the ogg data there is opus
```bash
$ ffmpeg -i out.ogg -f ffmetadata - 2>&1 | grep -A 1 "Stream #0:0"
  Stream #0:0(eng): Audio: opus, 48000 Hz, stereo, fltp
    Metadata:
```
If it does not show `Audio: opus` then the file is not a valid ogg opus file

### Show the page size of the ogg file
```bash
$ ffprobe -show_packets -select_streams a out.ogg 2>&1 | grep -E "duration|size"
...
duration=960
duration_time=0.020000
size=147
duration=960
duration_time=0.020000
size=139
duration=960
duration_time=0.020000
size=184
...
```
This means the stream will send every 20ms audio data. Mostly WebRTC is using 20ms audio data chunks.
