
win_size = 0.01;
fft_overlap = 0.5;

myDir = 'download/wav/'; 
myFiles = dir(fullfile(myDir,'*.wav'));
for k = 1:length(myFiles)
  baseFileName = myFiles(k).name;
  fullFileName = fullfile(myDir, baseFileName);
  fprintf(1, 'Now reading %s\n', fullFileName);
  
  [signal, Fs] = audioread(fullFileName);
  signal = signal(:,1); 
  hop_size = Fs*win_size;
  nfft = hop_size/fft_overlap;
  noverlap = nfft-hop_size;
  w = sqrt(hann(nfft));
  spectrogram(signal, w ,noverlap, nfft, Fs, 'yaxis' );
  colormap jet;
  axis off;
  colorbar('off');
  path = join(['spectrograms/',baseFileName,'.png']);
  path = erase(path, '.wav');
  saveas(gcf,path);
end