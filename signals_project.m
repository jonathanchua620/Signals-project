function [score, timeScore, timeAxis] = compareAudioFourier(referenceFile, testFile)
%COMPAREAUDIOFOURIER Compare two audio files using FFT-based time analysis.
%   Uses manual FFT() instead of spectrogram().

%% Read files
[ref, fsRef] = audioread("YNWA_og_acapella.mp4");
[test, fsTest] = audioread("YNWA_fan_acapella.mp4");

% Convert to mono
if size(ref,2) > 1, ref = mean(ref,2); end
if size(test,2) > 1, test = mean(test,2); end

% Resample if needed
if fsRef ~= fsTest
    test = resample(test, fsRef, fsTest);
end
fs = fsRef;

% Normalize
ref = ref / max(1e-12, max(abs(ref)));
test = test / max(1e-12, max(abs(test)));

% Trim silence
energyThresh = 1e-4;
ref = trimSilence(ref, energyThresh);
test = trimSilence(test, energyThresh);

%% Align signals using envelope cross-correlation
downFactor = max(1, round(fs/4000));
ref_ds  = ref(1:downFactor:end);
test_ds = test(1:downFactor:end);

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

% Match lengths
n = min(length(ref), length(test));
ref = ref(1:n);
test = test(1:n);

%% --- FFT PARAMETERS ---
winLen = 1024;
hop = winLen/4;
nfft = 2048;
win = hann(winLen, 'periodic');

% Number of frames
numFrames = floor((n - winLen) / hop) + 1;

% Preallocate
Mref = zeros(nfft/2+1, numFrames);
Mtest = zeros(nfft/2+1, numFrames);

%% --- MANUAL FFT PER FRAME ---
for k = 1:numFrames
    idx = (k-1)*hop + 1;
    frameRef = ref(idx:idx+winLen-1) .* win;
    frameTest = test(idx:idx+winLen-1) .* win;

    % FFT
    Fref = fft(frameRef, nfft);
    Ftest = fft(frameTest, nfft);

    % Magnitude (positive freqs only)
    Mref(:,k) = log1p(abs(Fref(1:nfft/2+1)));
    Mtest(:,k) = log1p(abs(Ftest(1:nfft/2+1)));
end

%% Frequency weighting
freqs = linspace(0, fs/2, nfft/2+1)';
freqWeight = 1 ./ (1 + (freqs/2000).^2);

Mref = Mref .* freqWeight;
Mtest = Mtest .* freqWeight;

%% --- TIME-RESOLVED SIMILARITY ---
timeScore = zeros(1, numFrames);

for k = 1:numFrames
    a = Mref(:,k) - mean(Mref(:,k));
    b = Mtest(:,k) - mean(Mtest(:,k));
    cs = dot(a,b) / (norm(a)*norm(b) + eps);
    timeScore(k) = max(0, (cs + 1)/2);
end

timeAxis = ((0:numFrames-1) * hop) / fs;

%% --- GLOBAL SIMILARITY ---
v1 = Mref(:) - mean(Mref(:));
v2 = Mtest(:) - mean(Mtest(:));
cosSim = dot(v1,v2) / (norm(v1)*norm(v2) + eps);
score = max(0, (cosSim + 1)/2);

%% --- PLOT ---
figure;
plot(timeAxis, timeScore, 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Similarity');
title('Time-Resolved FFT-Based Spectral Similarity');
ylim([0 1]);
grid on;

end

%% --- Helper function ---
function y = trimSilence(x, thresh)
frame = 512; hop = 256;
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
