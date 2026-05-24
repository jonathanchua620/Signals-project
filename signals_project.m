function score = compareAudioFourier(referenceFile, testFile)
%COMPAREAUDIOFOURIER Compare two audio files using Fourier-time analysis.
%   score = compareAudioFourier(referenceFile, testFile) reads the two audio
%   files, aligns them in time (simple cross-correlation), computes a
%   short-time Fourier transform (STFT) magnitude spectrogram for each,
%   and returns a similarity score in [0,1] where 1 indicates identical
%   magnitude spectrograms (after alignment) and 0 indicates no similarity.
%
%   Inputs:
%     referenceFile - path to reference audio (string)
%     testFile      - path to test audio (string)
%
%   Output:
%     score - similarity score between 0 and 1

% Read files
[ref, fsRef] = audioread(referenceFile);
[test, fsTest] = audioread(testFile);

% Convert to mono if needed
if size(ref,2) > 1, ref = mean(ref,2); end
if size(test,2) > 1, test = mean(test,2); end

% Resample if sample rates differ
if fsRef ~= fsTest
    test = resample(test, fsRef, fsTest);
    fsTest = fsRef;
end
fs = fsRef;

% Normalize amplitudes
ref = ref / max(1e-12, max(abs(ref)));
test = test / max(1e-12, max(abs(test)));

% Trim leading/trailing silence (simple energy threshold)
energyThresh = 1e-4;
ref = trimSilence(ref, energyThresh);
test = trimSilence(test, energyThresh);

% Align signals by cross-correlation on a downsampled envelope to be robust
downFactor = max(1, round(fs/4000)); % target ~4kHz for speed

% Toolbox-free decimation
ref_ds  = ref(1:downFactor:end);
test_ds = test(1:downFactor:end);

% Envelope extraction
envRef  = abs(hilbert(ref_ds));
envTest = abs(hilbert(test_ds));

[c, lags] = xcorr(envTest, envRef);
[~, idx] = max(c);
lagSamples = lags(idx) * downFactor;

if lagSamples > 0
    test = test(lagSamples+1:end);
else
    ref = ref(-lagSamples+1:end);
end
% Truncate to common length
n = min(length(ref), length(test));
ref = ref(1:n);
test = test(1:n);

% Parameters for STFT
winLen = 1024;          % window length
hop = winLen/4;         % 75% overlap
nfft = 2048;

% Compute magnitude spectrograms (log-magnitude)
[Sref, F, T] = spectrogram(ref, hann(winLen,'periodic'), winLen-hop, nfft, fs);
[Stest, ~, ~] = spectrogram(test, hann(winLen,'periodic'), winLen-hop, nfft, fs);
Mref = log1p(abs(Sref));
Mtest = log1p(abs(Stest));

% Ensure same time frames (they should, since signals same length)
m = min(size(Mref,2), size(Mtest,2));
Mref = Mref(:,1:m);
Mtest = Mtest(:,1:m);

% Optionally apply frequency weighting (more weight to lower freq)
freqWeight = 1 ./ (1 + (F/2000).^2); % emphasize below ~2kHz
Mref = bsxfun(@times, Mref, freqWeight);
Mtest = bsxfun(@times, Mtest, freqWeight);

% Compute similarity: normalized cosine similarity across flattened spectrograms
v1 = Mref(:);
v2 = Mtest(:);
v1 = v1 - mean(v1); v2 = v2 - mean(v2);
num = dot(v1,v2);
den = norm(v1)*norm(v2) + eps;
cosSim = num/den;

% Map cosine similarity [-1,1] to [0,1]
score = max(0, (cosSim + 1)/2);

end

function y = trimSilence(x, thresh)
% Remove leading/trailing sections where short-time RMS < thresh
frame = 512;
hop = 256;
sigLen = length(x);
nFrames = max(1,floor((sigLen-frame)/hop)+1);
rmsVals = zeros(nFrames,1);
    for k=1:nFrames
        i = (k-1)*hop+1;
        win = x(i:min(i+frame-1,sigLen));
        rmsVals(k) = sqrt(mean(win.^2));
    end
mask = rmsVals > thresh;
    if ~any(mask)
        y = x; return
    end
firstF = find(mask,1,'first');
lastF = find(mask,1,'last');
startS = max(1, (firstF-1)*hop+1);
endS = min(sigLen, (lastF-1)*hop+frame);
y = x(startS:endS);
end 
compareAudioFourier('YNWA_og_acapella.mp4', 'YNWA_fan_acapella.mp4') 