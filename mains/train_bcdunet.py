from __future__ import division
import sys
sys.path.append("../INF8225-Project/")

from architectures.BCDU_net.model.BCDU_net import BCDU_net
from architectures.BCDU_net.model.VisualizePredictions import *
from keras.optimizers import Adam
from keras.utils import plot_model
import pandas as pd
from keras.layers import *
from keras import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from architectures.BCDU_net.model.Evaluation import *
import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def run_training(args):
    retina_blood_vessel_dataset = RetinaBloodVesselDataset()
    preprocessing = Preprocessing()
    train_inputs, train_gt, train_bm = retina_blood_vessel_dataset.get_training_data()

    train_prepro_inputs, train_prepro_bm = preprocessing.run_preprocess_pipeline(train_inputs, "train", train_gt)
    #Using the einstein sum
    train_prepro_inputs = np.einsum('klij->kijl', train_prepro_inputs)
    train_prepro_bm = np.einsum('klij->kijl', train_prepro_bm)
    input_shape = (64,64,1)
    input = Input(input_shape)
    BCDU_NET = BCDU_net(input_shape)
    output = BCDU_NET(input)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer=Adam(lr=1e-7), loss='binary_crossentropy', metrics=['accuracy'])
    file = args.bcdu_models_dir + "BCDU_NET_architecture_.png"
    plot_model(model, to_file=file, show_shapes=True, show_layer_names=True)
    model.summary()
    file = args.bcdu_models_dir + "weights_lstm_.hdf5"
    mcp_save = ModelCheckpoint(file, save_best_only=True, monitor='val_loss', mode='min')
    reduce_LR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-7, mode='min')
    file = args.bcdu_models_dir + "history_log_.csv"
    csv_logger = CSVLogger(file, append=True)
    history = model.fit(train_prepro_inputs, train_prepro_bm,
              batch_size=8,
              epochs=50,
              shuffle=True,
              verbose=1,
              validation_split=0.2, callbacks=[csv_logger, mcp_save, reduce_LR])
    file = args.bcdu_models_dir + "model_50_epochs_.h5"
    model.save(file)
    file = args.bcdu_models_dir + "history_log_.csv"
    history = pd.read_csv(file, usecols=["epoch", "accuracy", "loss", "val_accuracy", "val_loss"])
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    file = args.bcdu_models_dir + "accuracy_history__.png"
    plt.savefig(file)

    # Plot training & validation loss values
    plt.clf()
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    file = args.bcdu_models_dir + "loss_history__.png"
    plt.savefig(file)


def run_eval(args):
    # EVALUATE THE MODEL
    evaluation = Evaluation()
    input_shape = (64, 64, 1)
    input = Input(input_shape)
    BCDU_NET = BCDU_net(input_shape)
    output = BCDU_NET(input)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer=Adam(lr=1e-7), loss='binary_crossentropy', metrics=['accuracy'])
    test_prepro_inputs, test_prepro_bm, new_h, new_w = evaluation.evaluation_data()
    file = args.bcdu_models_dir + "weights_lstm_.hdf5"
    model.load_weights(file)
    # # model.load_weights('../architectures/BCDU_net/BCDU_models/weight_lstm.hdf5')

    preds = model.predict(test_prepro_inputs, batch_size=16, verbose=1)
    predictions = np.einsum('kijl->klij', preds)
    np.save('50_predictions_', predictions)
    print("predicted images size :")
    print(predictions.shape)
    #Create an instance of the prediction visualizer class
    visualize_predictions = VisualizePredicitons()
    images, predictions_images, gt = visualize_predictions.make_visualizable(predictions, new_h, new_w, evaluation, test_prepro_bm)
    print("\n\n------Evaluation in progress--------")
    y_scores, y_true = visualize_predictions.field_of_view(predictions_images, gt, evaluation.test_bm)
    print(y_scores.shape)

    # # Get evaluation metrics
    evaluation.evaluation_metrics(y_true, y_scores)
    plt.clf()
    file = args.bcdu_tests_dir + "img_15_.png"
    plt.imshow((np.squeeze(images[14])), cmap="hot")
    plt.savefig(file)
    plt.clf()
    plt.imshow(np.squeeze(gt[14]), cmap='gray')
    file = args.bcdu_tests_dir + "gt_15_.png"
    plt.savefig(file)
    plt.clf()
    file = args.bcdu_tests_dir + "pred_img_15_.png"
    plt.imshow(np.squeeze(predictions_images[14]), cmap='gray')
    plt.savefig(file)

    # Visualize
    fig, ax = plt.subplots(10, 3, figsize=[100, 100])

    for idx in range(10):
        ax[idx, 0].imshow(np.squeeze((images[idx])), cmap="hot")
        ax[idx, 1].imshow(np.squeeze(gt[idx]), cmap='gray')
        ax[idx, 2].imshow(np.squeeze(predictions_images[idx]), cmap='gray')
    file = args.bcdu_tests_dir + "50_results_.png"
    plt.savefig(file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bcdu_models_dir', default='architectures/BCDU_net/BCDU_models/', type=str)
    parser.add_argument('--bcdu_tests_dir', default='architectures/BCDU_net/Tests/', type=str)
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--eval', default=False, action='store_true')
    args = parser.parse_args()

    if args.train:
        run_training(args)

    if args.eval:
        run_eval(args)
