#!/usr/bin/env python3

import numpy as np
import scipy
import tempfile
import os
import pathlib
import sys
import subprocess

# global
ax = None
begin = 0
take = 20
normalize = False
denoise = False
lowpass = 0
crop = False
quiet = False
loglevel = 32
scalefactor = False
DIVISION = 5
FACTOR = 0.75

ffmpegwav = 'ffmpeg -loglevel %s -ss %s -i "{}" %s -map 0:a:0 -c:a pcm_s16le -ac 1 -f wav "{}"'

audio_filters = {
    'default': 'atrim=0:%s,aresample=%s',
    'lowpass': 'lowpass=f=%s',
    'denoise': 'afftdn=nr=24:nf=-25'
}


def z_score_normalization(array):
    mean = np.mean(array)
    std_dev = np.std(array)
    normalized_array = (array - mean) / std_dev
    return normalized_array


def header(cmdstr):
    hdr = '-' * len(cmdstr)
    print('%s\n%s\n%s' % (hdr, cmdstr, hdr))


def get_max_rate(in1, in2):
    probe_audio = 'ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate -of default=noprint_wrappers=1'.split()
    command = probe_audio
    rates = []
    for file in [in1, in2]:
        cmdlist = command + [file]
        cmdstr = ' '.join(cmdlist)
        if not quiet:
            header(cmdstr)
        result = subprocess.run(cmdlist,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)
        if result.returncode == 0:
            if not quiet:
                print(result.stdout)
            rates.append(eval(result.stdout.split('=')[1]))
        else:
            print('FAIL in:\n', cmdstr)
            print(result.stderr)
            exit(1)
    return max(rates)


def get_duration(file):
    probe_audio = 'ffprobe -v error -select_streams a:0 -show_entries format=duration -of default=noprint_wrappers=1'.split()
    command = probe_audio
    cmdlist = command + [file]
    cmdstr = ' '.join(cmdlist)
    if not quiet:
        header(cmdstr)
    result = subprocess.run(cmdlist,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True)
    if result.returncode == 0:
        if not quiet:
            print(result.stdout)
        return eval(result.stdout.split('=')[1])
    else:
        print('FAIL in:\n', cmdstr)
        print(result.stderr)
        exit(1)


def in_out(command, infile, outfile):
    cmdstr = command.format(infile, outfile)
    if not quiet:
        header(cmdstr)
    ret = os.system(cmdstr)
    if 0 != ret:
        sys.exit(ret)


def get_sample(infile, rate, begin):
    outname = pathlib.Path(infile).stem + '_sample'
    with tempfile.TemporaryDirectory() as tempdir:
        outfile = pathlib.Path(tempdir) / (outname)
        filters = [audio_filters['default'] % (take, rate)]
        if int(lowpass):
            filters.append(audio_filters['lowpass'] % lowpass)
        if denoise:
            filters.append(audio_filters['denoise'])
        filter_string = '-af "' + ', '.join(filters) + '"'
        in_out(ffmpegwav % (loglevel, begin, filter_string), infile, outfile)
        r, s = scipy.io.wavfile.read(outfile)
        return s


def get_nsamples(file):
    duration = get_duration(file)
    return int(duration // take // DIVISION) - 1


def corrabs(s1, s2):
    ls1 = len(s1)
    ls2 = len(s2)
    padsize = ls1 + ls2 + 1
    padsize = 2**(int(np.log(padsize) / np.log(2)) + 1)
    s1pad = np.zeros(padsize)
    s1pad[:ls1] = s1
    s2pad = np.zeros(padsize)
    s2pad[:ls2] = s2
    corr = scipy.fft.ifft(scipy.fft.fft(s1pad) * np.conj(scipy.fft.fft(s2pad)))
    ca = np.absolute(corr)
    xmax = np.argmax(ca)
    return ls1, ls2, padsize, xmax, ca


def removeOutliers(inputx, inputy):
    x = np.array(inputx)
    y = np.array(inputy)

    # Ajuste de regresión lineal
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    y_pred = slope * x + intercept  # Valores predichos

    # Calcular residuos
    residuals = y - y_pred
    std_res = np.std(residuals)

    # Filtrar datos usando 2 desviaciones estándar como umbral
    threshold = FACTOR * std_res
    filtered_indices = np.abs(residuals) < threshold
    x_filtered, y_filtered = x[filtered_indices], y[filtered_indices]

    # Volver a calcular la regresión con los datos filtrados
    slope_f, intercept_f, _, _, _ = scipy.stats.linregress(x_filtered, y_filtered)
    return x_filtered, y_filtered, slope_f


def cli_parser(**ka):
    import argparse
    parser = argparse.ArgumentParser(
        prog='syncstart',
        description=file_offset.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--version', action='version', version="1.1.1")

    if 'in1' not in ka:
        parser.add_argument(
            'in1',
            help='First media file to sync with second.')
    if 'in2' not in ka:
        parser.add_argument(
            'in2',
            help='Second media file to sync with first.')
    if 'begin' not in ka:
        parser.add_argument(
            '-b', '--begin',
            dest='begin',
            action='store',
            default=0,
            help='Begin comparison X seconds into the inputs. (default: 0)')
    if 'take' not in ka:
        parser.add_argument(
            '-t', '--take',
            dest='take',
            action='store',
            default=20,
            help='Take X seconds of the inputs to look at. (default: 20)')
    if 'scalefactor' not in ka:
        parser.add_argument(
            '-s', '--scalefactor',
            dest='scalefactor',
            action='store_true',
            default=False,
            help='Calculate scale factor')
    if 'normalize' not in ka:
        parser.add_argument(
            '-n', '--normalize',
            dest='normalize',
            action='store_true',
            default=False,
            help='Normalizes audio values from each stream.')
    if 'denoise' not in ka:
        parser.add_argument(
            '-d', '--denoise',
            dest='denoise',
            action='store_true',
            default=False,
            help='Reduces audio noise in each stream.')
    if 'lowpass' not in ka:
        parser.add_argument(
            '-l', '--lowpass',
            dest='lowpass',
            action='store',
            default=0,
            help="Audio option: Discards frequencies above the specified Hz, \
            e.g., 300. 0 == off (default)")
    if 'crop' not in ka:
        parser.add_argument(
            '-c', '--crop',
            dest='crop',
            action='store_true',
            default=False,
            help='Video option: Crop to 4:3. Helpful when aspect ratios differ.')
    if 'quiet' not in ka:
        parser.add_argument(
            '-q', '--quiet',
            dest='quiet',
            action='store_true',
            default=False,
            help='Suppresses standard output except for the CSV result.\
            Output will be: file_to_advance, seconds_to_advance')
    return parser


def file_offset(**ka):
    """CLI interface to sync two media files using their audio streams.
    ffmpeg needs to be available.
    """

    parser = cli_parser(**ka)
    args = parser.parse_args().__dict__
    ka.update(args)

    global begin, take, normalize, scalefactor, denoise, lowpass, crop, quiet, loglevel
    in1, in2, begin, take = ka['in1'], ka['in2'], ka['begin'], ka['take']
    normalize, scalefactor, denoise, lowpass = ka['normalize'], ka['scalefactor'], ka['denoise'], ka['lowpass']
    loglevel = 16 if quiet else 32

    sr = get_max_rate(in1, in2)
    # s1, s2 = get_sample(in1, sr), get_sample(in2, sr)
    if scalefactor:
        nsamples = get_nsamples(in1)
        offsets = []
        intervals = list(range(0, nsamples * take * DIVISION, take * DIVISION))
        for i in range(nsamples):
            _, _, padsize, xmax, _ = corrabs(get_sample(in1, sr, i * take * DIVISION),
                                             get_sample(in2, sr, i * take * DIVISION))
            offset = (padsize - xmax) / sr if xmax > padsize // 2 else xmax * -1 / sr
            offsets.append(offset)
        intervalsout, offsetsout, slope = removeOutliers(intervals, offsets)
        if slope < 0:
            atempo = 1 / (1 + slope)
        else:
            atempo = 1 - slope
        offset = np.average((offsetsout + intervalsout) * atempo - intervalsout) * -1
        sync_text = "Factor = %s   |   offset = %s ms"
        print(sync_text % (atempo, offset))


    else:
        s1, s2 = get_sample(in1, sr), get_sample(in2, sr)
        if normalize:
            s1, s2 = z_score_normalization(s1), z_score_normalization(s2)
        ls1, ls2, padsize, xmax, ca = corrabs(s1, s2)
        sync_text = """
    ==============================================================================
    %s needs 'ffmpeg -ss %s' cut to get in sync
    ==============================================================================
    """
        file = in2
        offset = (padsize - xmax) / sr
        if xmax < padsize // 2:
            offset = xmax * -1 / sr
        if not quiet:  # default
            print(sync_text % (file, offset))
        else:  # quiet
            # print csv: file_to_advance, seconds_to_advance
            print("%s, %s" % (file, offset))
        return file, offset


main = file_offset
if __name__ == '__main__':
    main()
