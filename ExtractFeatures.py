class feature_extractor:
  def __init__(self,datadir,savedir,song_speech ='02',splitnotsplit = True):
    self.data   = {}
    self.sdir   = savedir
    self.dir    = datadir
    self.sns    = splitnotsplit
    self.sp     = song_speech
    self.labels = {'01':['neutral',0],'02':['calm',1],'03':['happy',2],'04':['sad',3],'05':['angry',4],'06':['fearful',5],'07':['disgust',6],'08':['surprised',7]}


  def navigate(self,):
    ''' 
    ** Only with sound files
    This takes:
    1- Data directory: where to navigate to find audio files
    2- song_speech   : 02 to extract melspectogram from song audio files/ 01 to extract them from speech audio files

    This returns/formulates:
    1- A dictionary holds each melspectogram directory with its label/metadata 
    '''
    key = None
    for dirname, _, filenames in os.walk(self.dir):
      for filename in filenames:
        meta = self._splitter(filename)
        if (meta is None): continue

        meta['Dir'] = os.path.join(dirname,filename)

        key         = str (uuid.uuid1()).split('-')[0]
        self.data[key] = meta 

  def _splitter(self,filename,):
    '''
    this decodes file name into:
    1- VocalChannel 
    2- Class [number from 0 to 7]
    3- Intensity level: 0 for non intense -- 1 for intense
    4- WhichStatement : 0 for kids -- 1 for dogs
    5- Is it a repeated record: 0 for no -- 1 for yes
    6- Gender: 0 for male -- 1 for female
    '''
    codes = filename.split('.')[0].split('-')

    if (codes[1] != self.sp):
      return None
    
    meta = {}
    meta['VocalChannel']   = 'song' if codes[1] == self.sp else 'speech'
    meta['Class']          = self.labels[codes[2]]
    meta['IntenseVoice']   = 0 if codes[3] == '01' else 1
    meta['WhichStatement'] = 0 if codes[4] == '01' else 1
    meta['Repeated?']      = 0 if codes[5] == '01' else 1
    meta['gender']         = 0 if (int(codes[6])%2 != 0) else 1
  
    return meta
  
  def extract(self,removeaudio=False,whichto = 'both',chunk=3):
    '''
    takes: 
    1-removeaudio: True in case deletion of audio file after extracting features
    2-whichto    : 'both' for melspec+mfccs -- 'mel' for mel only -- 'mfcc' for mfccs only
    3-chunk      : chunk of audio file to split
    '''
    audio,sr = None,None   
    for key in tqdm(self.data.keys()):
      audio,sr,hop_length,win_length  = self._process_cough_file(self.data[key]['Dir'],self.sns,removeaudio,chunk=chunk)

      if (whichto =='mel'):
        self.data[key]['Dir']  = self._get_melspec(self.data[key]['Dir'],audio,sr,key)

      elif (whichto == 'mfcc'):
        self.data[key]['Mfcc'] = self._get_MFCCS(audio,sr,hop_length,win_length)

      else:
        self.data[key]['Dir']  = self._get_melspec(self.data[key]['Dir'],audio,sr,key)
        self.data[key]['Mfcc'] = self._get_MFCCS(audio,sr,hop_length,win_length)

  def _process_cough_file(self,path, splitnotsplit,removeaudio=False ,chunk=3):
    x,sr = librosa.load(path, sr=48000)        
    if len(x)/sr < 0.3 or len(x)/sr > 30:
        return None,None
    
    if removeaudio:
      os.remove(path)

    if (splitnotsplit == False):
      return x,sr
    
    x = self._trim_silence(x, pad=0.25*sr, db_max=50) 
    x = x[:np.floor(chunk*sr).astype(int)]
    
    #pads to chunk size if smaller
    x_pad = np.zeros(int(sr*chunk))
    x_pad[:min(len(x_pad), len(x))] = x[:min(len(x_pad), len(x))]
    
    hop_length = np.floor(0.010*sr).astype(int) #10ms
    win_length = np.floor(0.020*sr).astype(int) #32ms
    return x_pad,sr,hop_length,win_length

  # Trims leading and trailing silence
  def _trim_silence(self,x, pad=0, db_max=50):
    _, ints = librosa.effects.trim(x, top_db=db_max, frame_length=256, hop_length=64)
    start   = int(max(ints[0]-pad, 0))
    end     = int(min(ints[1]+pad, len(x)))
    return x[start:end]

  def _get_chromaCQT(self,dir,audio,sr,name):
    plt.ioff()
    fig    = plt.figure()
    chroma = librosa.feature.chroma_cqt(y=audio,sr=sr)
    librosa.display.specshow(chroma)
    fig.canvas.draw()
    chroma = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    chroma = chroma.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig=fig)
    #chroma = chroma[80:250,80:300]

    savepath = os.path.join(self.sdir,name+'.png')
    cv2.imwrite(savepath,chroma)

    return savepath

  def _get_melspec(self,dir,audio,sr,name):
    #Mel Spectogram
    plt.ioff()
    fig      = plt.figure()
    melspec  = librosa.feature.melspectrogram(y=audio,sr=sr)
    s_db     = librosa.power_to_db(melspec, ref=np.max)
    librosa.display.specshow(s_db)
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig=fig)
    #img = img[80:250,80:300]

    savepath = os.path.join(self.sdir,name+'.png')
    cv2.imwrite(savepath,img)

    return savepath

    
  def _get_MFCCS(self,audio,sr,hop_length,win_length,final_dim=(300,200)):
    #For MFCCS 
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mels=200, n_mfcc=200, n_fft=2048, 
                                hop_length=hop_length)
    mfcc = np.swapaxes(mfcc, 0, 1)
    mfcc = mfcc[:final_dim[0], :final_dim[1]]
    return mfcc



if __name__ == '__main__':
    data_dir = ''
    Master = feature_extractor(data_dir,savedir='/content/melspecs')
    Master.navigate()
    Master.extract()