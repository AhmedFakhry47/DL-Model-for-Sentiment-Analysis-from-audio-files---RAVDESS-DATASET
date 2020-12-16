
class Evaluation(keras.callbacks.Callback):
  def __init__(self, val_data_gen, val_labels, test_data_gen, test_labels):
    super(Callback, self).__init__()
    self.test_data = test_data_gen
    self.val_labels = val_labels
    self.val_data = val_data_gen
    self.test_labels = test_labels

  def on_epoch_end(self, epoch, logs=None):
    y_preds = self.model.predict_generator(self.val_data)
    print(' | val_auc:', roc_auc_score(self.val_labels[:len(y_preds)], y_preds))

    y_preds = self.model.predict_generator(self.test_data)
    print(' | test_auc:', roc_auc_score(self.test_labels[:len(y_preds)], y_preds))


def MODEL_RESNET50(input_shape=(224,224,3),n_classes=80,act='softmax',weights=None,loss_weight=None):
  base_model = resnet50.ResNet50(input_shape=input_shape,weights=weights, include_top=False)
  x1         = keras.layers.GlobalAveragePooling2D()(base_model.output)
  x2         = keras.layers.GlobalMaxPooling2D()(base_model.output)
  x          = concatenate([x1,x2])

  x = BatchNormalization()(x)
  x = Dropout(0.5)(x)
  
  x = Dense(256, activation='relu')(x)
  x = BatchNormalization()(x)
  x = Dropout(0.5)(x)

  output = keras.layers.Dense(n_classes, activation=act)(x)
  RESNET = keras.models.Model(inputs=[base_model.input], outputs=[output])

  lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=1000,
    decay_rate=0.8)
  optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
  RESNET.compile(optimizer='Adam',loss='CategoricalCrossentropy',
                 metrics=['CategoricalAccuracy',tf.keras.metrics.AUC(multi_label=True)])

  return RESNET

