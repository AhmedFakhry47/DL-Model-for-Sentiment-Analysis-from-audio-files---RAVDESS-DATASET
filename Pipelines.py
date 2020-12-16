class ExtrPipeline(tf.keras.utils.Sequence):
  '''
  Pipeline for MFCC
  '''
  def __init__(self,imgfiles,labels,batch_size,target_size=(64,64),shuffle=False,scale=255,n_classes=2,n_channels=1):
    self.batch_size = batch_size
    self.dim        = target_size
    self.labels     = labels
    self.imgfiles   = imgfiles
    self.n_classes  = n_classes
    self.shuffle    = shuffle
    self.n_channels = n_channels
    self.scale      = scale
    
    self.on_epoch_end()
  
  def __len__(self):
    return int(np.floor(len(self.imgfiles) / self.batch_size))

  def __getitem__(self, index):
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    X, y = self.__data_generation(indexes)
    return X, y

  def on_epoch_end(self):
    self.indexes = np.arange(len(self.imgfiles))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)
  
  def __data_generation(self, list_IDs_temp):
    X = np.empty((self.batch_size, *self.dim, self.n_channels))
    y = np.empty((self.batch_size,self.n_classes))

    # Generate data
    for i, ID in enumerate(list_IDs_temp):
      #Mel Spectogram
      X[i,] = self.imgfiles[ID][:,:,np.newaxis]
      
      # Store class
      y[i]  = self.labels[ID]
    
    return X, y


class image_aug:
  def vertical_shift(self,img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = h*ratio
    if ratio > 0:
        img = img[:int(h-to_shift), :, :]
    if ratio < 0:
        img = img[int(-1*to_shift):, :, :]
    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img

  def brightness(self,img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

  def horizontal_shift(self,img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = w*ratio
    if ratio > 0:
        img = img[:, :int(w-to_shift), :]
    if ratio < 0:
        img = img[:, int(-1*to_shift):, :]
    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img
    
  def channel_shift(self,img, value):
    value = int(random.uniform(-value, value))
    img = img + value
    img[:,:,:][img[:,:,:]>255]  = 255
    img[:,:,:][img[:,:,:]<0]  = 0
    img = img.astype(np.uint8)
    return img
    
  def rotation(self,img, angle):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img
  
  def zoom(self,img, value):
    if value > 1 or value < 0:
        print('Value for zoom should be less than 1 and greater than 0')
        return img
    value = random.uniform(value, 1)
    h, w = img.shape[:2]
    h_taken = int(value*h)
    w_taken = int(value*w)
    h_start = random.randint(0, h-h_taken)
    w_start = random.randint(0, w-w_taken)
    img = img[h_start:h_start+h_taken, w_start:w_start+w_taken, :]
    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img
    
class MasterPipeline(tf.keras.utils.Sequence):
  def __init__(self,imgfiles,labels,batch_size,target_size=(64,64),shuffle=False,scale=255,n_classes=2,n_channels=3,augment=False,aseed=[650,250]):
    self.batch_size = batch_size
    self.dim        = target_size
    self.labels     = labels
    self.imgfiles   = imgfiles
    self.n_classes  = n_classes
    self.shuffle    = shuffle
    self.n_channels = n_channels
    self.scale      = scale
    
    self.augment    = augment
    self.augmentinfo= aseed

    self.on_epoch_end()
  
  def __augment(self,img):
    key = random.randint(0,8)
    augmenter = image_aug()
    aug_img   = None
    return img

    if (key == 0):
      aug_img = augmenter.brightness(img,random.randint(0,100))
    elif (key == 1):
      aug_img = augmenter.rotation(img,random.randint(0,360))
    elif (key == 2):
      aug_img = augmenter.horizontal_shift(img,random.random())
    elif (key == 3):
      aug_img = augmenter.zoom(img,random.random())
    elif (key == 4):
      aug_img = augmenter.vertical_shift(img,random.random())
    else:
      aug_img = img

    del augmenter
    return np.array(aug_img)

  def __len__(self):
    return int(np.floor(len(self.imgfiles) / self.batch_size))

  def __getitem__(self, index):
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    X, y = self.__data_generation(indexes)
    return X, y

  def on_epoch_end(self):
    self.iaa        = 0
    self.aseed      = [random.randint(0,self.augmentinfo[0]) for _ in range(self.augmentinfo[1])]

    self.indexes = np.arange(len(self.imgfiles))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)
  
  def __data_generation(self, list_IDs_temp):
    X = np.empty((self.batch_size, *self.dim, self.n_channels))
    y = np.empty((self.batch_size,self.n_classes))

    # Generate data
    for i, ID in enumerate(list_IDs_temp):
      #Mel Spectogram
      img   = cv2.imread(self.imgfiles[ID])
      img   = cv2.resize(img,tuple(reversed(self.dim)),interpolation = cv2.INTER_CUBIC)
      X[i,] = img / self.scale
      
      # Store class
      y[i]  = self.labels[ID]
    
    if ((self.augment == True) and (self.iaa in self.aseed)):
      X = np.array(list(map(self.__augment, X))).astype(np.float32)
    
    self.iaa += 1
    return X, y