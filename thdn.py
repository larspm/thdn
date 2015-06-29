# This script tries to calculate Total Harmonic Distortion plus Noise (THD+N).
# It takes as input a .wav file, which must contain a sequence of spectrally
# stationary time intervals. Each interval should be at lest a couple of seconds
# long, or whatever. The change in frequency should not be very small.
#
# The script will try to find the stationary intervals. THD+N is then calculated
# for each interval.
#
# When calculating THD+N, it is assumed that the original signal x, is a pure
# sinusoid of some frequency, phase and amplitude,
#
#         x(t) = A*sin(w0*t+p),
#
# and that it has gone through some kind memoryless nonlinearity H, distorting it and
# adding noise, producing an output signal y given by
#
#         y(t) = H(x(t))+n(t).
#
# Since H is time invariant and memoryless, H(x(t)) can be composed of sinusoids of
# frequencies that are integer multiples of the fundamental:
#
#         y(t) = sum(a[k]*sqrt(2)*sin(w0*k*t+p[k]), k=1..∞)  + n(t).
#
# The sqrt(2) is there to make the amplitudes RMS amplitudes.
#
# THD+N is defined as sqrt(sum(a[k]^2, k=2..∞) + N^2)/a[1], where N is the noise RMS.
# It is fairly easy to calculate that RMS[y] = sqrt(norm(a) + MS[n]), which means that
# if we can subtract the fundamental from y and calculate the RMS, we have the THD+N
# numerator.
#
# If we can find the fundamental frequency, w0, and we choose a portion of the
# recording that contains a whole number of fundamental cycles, the fundamental
# will be orthogonal to the harmonics, and the true (a[1],p[1]) are ones that
# minimize rms(y(t)-a[1]*sin(w0*t+p[1])). We will assume that a recording
# containing a large number of cycles will be orthogonal enough to get good results.
#
# Since the RMS of a vector is proportional to the (euclidean) norm, we will treat
# this as a linear algebra problem. With some awful abuse of notation, we want to
# minimize
#
#         norm(y(t) - a[1]*sin(w0*t+p[1])) = norm(y(t) - b[1]*sin(w0*t) - c[1]*cos(w0*t))
#
# The latter form has a closed form solution. Of course, we don't know w0, but
# we can search for the best w0 by calculating the above for each value of w0.
#
# Searching for the best w0 can take a lot of time. We will assume that the
# fundamental is the strongest frequency component. We will start by finding
# the strongest DFT component, and then do a search in a range around that
# component. When we find the frequency out of a uniform selection in that range
# that gives the best THD+N, we will zoom in around that value and do the same
# thing again. We continue this until the difference between the largest and smallest
# THD+N value goes below a certain threshold.
#
# The search for stationary intervals is done by dividing the signal into blocks,
# and comparing the absolute DFT spectrum of neighbor blocks. There are some
# problems with this approach. Taking an excerpt of a sinusoid smears the frequency
# content with a sinc. How the overlaps of the smearing of the positive and negative
# frequency components interferes constructively and destructively with each other
# is not independent of phase, i.e. two different excerpts of the same sinusoid
# may not have the same absolute DFT, especially for frequencies close to DC or
# the Nyquist frequency. This makes it hard to set a threshold for when to consider
# two blocks equal or not. Currently, some transitions are not detected, and some
# times false transitions are detected.
#
# TBD:
#    * Handling errors, corner cases, unexpected events, etc.
#    * Handle non-mono input
#    * Handle other sample bit sizes other than 16 bit.
#    * Better detection of frequency transitions.
#    * Option to do faster search by only searching for integer Hz fundamentals.
#    * Option to have THD+N relative to both fixed level and fundamental level.
#    * Output format
#    * Better interface

import numpy as np

def find_stationaries(fname):
    import wave

    wf = wave.open(fname)

    nchans = wf.getnchannels()
    R      = wf.getsampwidth()
    Fs     = wf.getframerate()
    N      = wf.getnframes()

    # support more bitwidths
    x = np.fromstring(wf.readframes(N), {2:np.dtype('<i2')}[R]) / 2**(8*R-1)

    K = 1024*4

    edge_trim = 3000
    curr_first = edge_trim
    while (curr_first+K) < (len(x)-edge_trim):
        X = abs(np.fft.fft(x[curr_first:curr_first+K]))
        NX = np.linalg.norm(X)
        k = 1
        while (curr_first+(k+1)*K) < (len(x)-edge_trim):
            X_ = abs(np.fft.fft(x[curr_first+(k)*K:curr_first+(k+1)*K]))
            if np.linalg.norm(X-X_)/NX > 0.2:
                break
            k = k + 1

        if k*K > Fs*0.5:
            yield (curr_first, curr_first+k*K, Fs, x[curr_first:(curr_first+k*K)])

        curr_first = curr_first+(k+1)*K

def thdn(x):

    N = len(x)
    n = np.mat(np.arange(N)).T

    X = abs(np.fft.fft(x))

    x=np.mat(x).T

    f0_init = X[:N//2].argmax() / N

    f_delta = 0.5/20
    f_start = max(f0_init-f_delta/2, 18/48000)
    f_stop  = min(f0_init+f_delta/2, 0.5)

    K=30

    thdn_min = 100000000
    f_opt = -1

    for dummy in range(20):
        thdn_max = 0

        res = np.zeros([K,2])
        for k in range(K):
            f = f_start + (f_stop - f_start)*k/K

            A = np.hstack([np.ones(n.shape), np.cos(2*np.pi*f*n), np.sin(2*np.pi*f*n)])

            c,E,_,_ = np.linalg.lstsq(A,x)
            y=A*c
            if E.size == 0:
                print(f)
                continue

            thdn_tmp = np.sqrt(float(E)) / np.linalg.norm(y)

            if thdn_tmp < thdn_min:
                thdn_min = thdn_tmp
                f_opt = f
                y_opt = y

            if thdn_tmp > thdn_max:
                thdn_max = thdn_tmp

            res[k,0] = f
            res[k,1] = thdn_tmp


        if 20*np.log10(thdn_max/thdn_min) < 1:
            break

        f_delta = (f_stop - f_start)/20
        f_start = max(f_opt-f_delta/2, 0.0)
        f_stop  = min(f_opt+f_delta/2, 0.5)

    return f_opt, thdn_min, y_opt

if __name__ == "__main__":
    import os,sys
    if not (len(sys.argv)>1 and os.path.isfile(sys.argv[1])):
        print('usage: {} file.wav'.format(sys.argv[0]))
        sys.exit(1)
    for first, last, Fs, x in find_stationaries(sys.argv[1]):
        f,th,y = thdn(x)
        print('[{},{}] f0={} thdn={} dB'.format(first/Fs,last/Fs,f*Fs,20*np.log10(th)))
