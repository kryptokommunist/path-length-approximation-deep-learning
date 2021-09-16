import sys
import os
import io
import time
import copy
import math
import pickle
import numpy as np
from utils import *

import torch
from torchsummary import summary
from torch.utils import data as torch_data
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

class Trainer():

    def __init__(self, graph_name, emb_dim, split, train_path, val_path, test_path, params):
        print("Split: {}".format(split))
        self.params = params
        self.path = "/run/output/{}/{}/{}/".format(graph_name, emb_dim, split)
        self.split = split
        self.scores = {}
        self.load_data(train_path, val_path, test_path)
        self.create_baseline()
        self.train_nn()
        
    def ensure_folders_exist(self, path):
        """
        Creates folders that do not exist yet on the given path
        """
        if not os.path.exists(os.path.dirname(path)):
            try:
                os.makedirs(os.path.dirname(path))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

    def load_data(self, train_path, val_path, test_path):
        if self.split == "(1.00, 0.0, 0.0)":
            test_path = train_path
            val_path = train_path
        self.x_train, self.y_train = pickle.load(open(train_path, 'rb'))
        self.x_cv, self.y_cv = pickle.load(open(val_path, 'rb'))
        self.x_test, self.y_test = pickle.load(open(test_path, 'rb'))
        print('shapes of train, validation, test data', self.x_train.shape, self.y_train.shape, self.x_cv.shape, self.y_cv.shape, self.x_test.shape, self.y_test.shape)
        values, counts = np.unique(self.y_train, return_counts=True)
        self.max_dist = max(values)
        self.num_features = self.x_train.shape[1]
        print('Frequency of distance values before sampling', values, counts)
        np.random.seed(999)
        self.x_train, self.y_train = unison_shuffle_copies(self.x_train, self.y_train)

    def create_baseline(self):
        baseline_model = LinearRegression(fit_intercept=True, normalize=True, n_jobs=-1).fit(self.x_train, self.y_train)
        y_pred = baseline_model.predict(self.x_test)
        y_class = np.round(y_pred)
        baseline_acc = accuracy_score(self.y_test, y_class)*100
        baseline_mse = mean_squared_error(self.y_test, y_pred)
        baseline_mae = mean_absolute_error(self.y_test, y_pred)
        baseline_mre = mean_absolute_percentage_error(self.y_test, y_pred)

        self.scores["baseline"] = {"acc": baseline_acc, "mse": baseline_mse, "mae": baseline_mae, "mre": baseline_mre}

        print("Baseline: Accuracy={}%, MSE={}, MAE={}".format(round(baseline_acc, 2), round(baseline_mse,2), round(baseline_mae,2)))

    def train_nn(self):
        params = self.params
        params['input_size'] = self.num_features
        device = "cuda" #torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        print("device:", device)

        trainset = torch_data.TensorDataset(torch.as_tensor(self.x_train, dtype=torch.float, device=device), torch.as_tensor(self.y_train, dtype=torch.float, device=device))
        train_dl = torch_data.DataLoader(trainset, batch_size=params['batch_size'], drop_last=True)

        val_dl = torch_data.DataLoader(torch_data.TensorDataset(torch.as_tensor(self.x_cv, dtype=torch.float, device=device), torch.as_tensor(self.y_cv, dtype=torch.float, device=device)), batch_size=params['batch_size'], drop_last=True)

        test_dl = torch_data.DataLoader(torch_data.TensorDataset(torch.as_tensor(self.x_test, dtype=torch.float, device=device), torch.as_tensor(self.y_test, dtype=torch.float, device=device)), batch_size=params['batch_size'], drop_last=True)

        print('value counts in whole data', np.unique(self.y_train, return_counts=True))
        count = 0
        for i, data in enumerate(train_dl, 0):
            input, target = data[0], data[1]
            t = torch.unique(target, return_counts=True)[1]
            if (t==params['batch_size']).any().item():
                count += 1
        print('{} ({}%) batches have all same targets'.format(count, np.round(count/len(train_dl)*100, 2) ))

        torch.manual_seed(9999)

        def get_model():
            """
            creates a PyTorch model. Change the 'params' dict above to
            modify the neural net configuration.
            """
            model = torch.nn.Sequential(
                torch.nn.Linear(params['input_size'], params['hidden_units_1']),
                torch.nn.BatchNorm1d(params['hidden_units_1']),
                # torch.nn.Dropout(p=params['do_1']),
                torch.nn.ReLU(),
                torch.nn.Linear(params['hidden_units_1'], params['hidden_units_2']),
                torch.nn.BatchNorm1d(params['hidden_units_2']),
                # torch.nn.Dropout(p=params['do_2']),
                torch.nn.ReLU(),
                torch.nn.Linear(params['hidden_units_2'], params['hidden_units_3']),
                torch.nn.BatchNorm1d(params['hidden_units_3']),
                # torch.nn.Dropout(p=params['do_3']),
                torch.nn.ReLU(),
                torch.nn.Linear(params['hidden_units_3'], params['output_size']),
                torch.nn.ReLU(),
                # torch.nn.Softplus(),
            )
            model.to(device)
            return model

        def poisson_loss(y_pred, y_true):
            """
            Custom loss function for Poisson model.
            Equivalent Keras implementation for reference:
            K.mean(y_pred - y_true * math_ops.log(y_pred + K.epsilon()), axis=-1)
            For output of shape (2,3) it return (2,) vector. Need to calculate
            mean of that too.
            """
            y_pred = torch.squeeze(y_pred)
            loss = torch.mean(y_pred - y_true * torch.log(y_pred+1e-7))
            return loss

        model = get_model()

        print('model loaded into device=', next(model.parameters()).device)

        # this is just to capture model summary as string
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        summary(model, input_size=(params['input_size'], ), device=device)

        sys.stdout = old_stdout
        model_summary = buffer.getvalue()
        print('model-summary\n', model_summary)
        # later this 'model-summary' string can be written to tensorboard

        lr_reduce_patience = 20
        lr_reduce_factor = 0.1

        loss_fn = poisson_loss
        # optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], momentum=0.9, dampening=0, weight_decay=0, nesterov=True)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=params['lr'], alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

        if params['lr_sched'] == 'reduce_lr_plateau':
            lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_reduce_factor, patience=lr_reduce_patience, verbose=True, threshold=0.00001, threshold_mode='rel', cooldown=0, min_lr=1e-9, eps=1e-08)
        elif params['lr_sched'] == 'clr':
            lr_sched = torch.optim.lr_scheduler.CyclicLR(optimizer, params['min_lr'], params['max_lr'], step_size_up=8*len(train_dl), step_size_down=None, mode=params['lr_sched_mode'], last_epoch=-1, gamma=params['gamma'])

        print('lr scheduler type:', lr_sched)
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        
        global lr_arr
        lr_arr = np.zeros((len(train_dl), ))

        def find_lr(init_value = 1e-8, final_value=10., beta = 0.98):
            global lr_arr
            num = len(train_dl)-1
            mult = (final_value / init_value) ** (1/num)
            lr = init_value
            optimizer.param_groups[0]['lr'] = lr
            avg_loss = 0.
            best_loss = 0.
            batch_num = 0
            losses = []
            log_lrs = []
            lrs = []
            for data in train_dl:
                batch_num += 1
                # As before, get the loss for this mini-batch of inputs/outputs
                inputs, labels = data
                # inputs, labels = Variable(inputs), Variable(labels)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                # Compute the smoothed loss
                avg_loss = beta * avg_loss + (1-beta) *loss.item()
                smoothed_loss = avg_loss / (1 - beta**batch_num)
                # Stop if the loss is exploding
                if batch_num > 1 and smoothed_loss > 4 * best_loss:
                    return log_lrs, losses
                # Record the best loss
                if smoothed_loss < best_loss or batch_num==1:
                    best_loss = smoothed_loss
                # Store the values
                losses.append(smoothed_loss)
                log_lrs.append(math.log10(lr))
                lrs.append(lr)
                lr_arr[batch_num-1] = lr
                # Do the SGD step
                loss.backward()
                optimizer.step()
                # Update the lr for the next step
                lr *= mult
                optimizer.param_groups[0]['lr'] = lr
            return log_lrs, losses

        lrs, losses = find_lr()
        print('returned', len(losses))
        plt.figure()
        plt.plot(lr_arr[:len(lrs)], losses)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.title('LR range plot')
        plt.xlabel('Learning rates')
        plt.ylabel('Losses')
        im_path = os.path.join(self.path, "lr range plot.png")
        self.ensure_folders_exist(im_path)
        fig, ax = plt.subplots()
        fig.savefig(im_path)

        def evaluate(model, dl):
            """
            This function is used to evaluate the model with validation.
            args: model and data loader
            returns: loss
            """
            model.eval()
            final_loss = 0.0
            count = 0
            with torch.no_grad():
                for data_cv in dl:
                    inputs, dist_true = data_cv[0], data_cv[1]
                    count += len(inputs)
                    outputs = model(inputs)
                    loss = loss_fn(outputs, dist_true)
                    final_loss += loss.item()
            return final_loss/len(dl)

        def save_checkpoint(state, state_save_path):
            if not os.path.exists("/".join(state_save_path.split('/')[:-1])):
                os.makedirs("/".join(state_save_path.split('/')[:-1]))
            torch.save(state, state_save_path)


        last_loss = 0.0
        min_val_loss = np.inf
        patience_counter = 0
        early_stop_patience = 50
        best_model = None
        train_losses = []
        val_losses = []

        tb_path = os.path.join(self.path, 'logs/runs')
        checkpoint_path = os.path.join(tb_path, 'checkpoints')
        resume_training = False
        start_epoch = 0
        iter_count = 0

        if os.path.exists(checkpoint_path):
            #raise Exception("this experiment already exists!")
            print("Already ran training on {}".format(checkpoint_path))
            return
        
        self.ensure_folders_exist(checkpoint_path)

        writer = SummaryWriter(log_dir=tb_path, comment='', purge_step=None, max_queue=1, flush_secs=30, filename_suffix='')
        writer.add_graph(model, input_to_model=torch.zeros(params['input_size'], device=device).view(1,-1), verbose=False)  # not useful

        # resume training on a saved model
        if resume_training:
            prev_checkpoint_path = '../outputs/logs/runs/run42_clr_g0.95/checkpoints'  # change this
            suffix = '1592579305.7273214'  # change this
            model.load_state_dict(torch.load(prev_checkpoint_path+'/model_'+suffix+'.pt'))
            optimizer.load_state_dict(torch.load(prev_checkpoint_path+'/optim_'+suffix+'.pt'))
            lr_sched.load_state_dict(torch.load(prev_checkpoint_path+'/sched_'+suffix+'.pt'))
            state = torch.load(prev_checkpoint_path+'/state_'+suffix+'.pt')
            start_epoch = state['epoch']
            writer.add_text('loaded saved model:', str(params))
            print('loaded saved model', params)

        writer.add_text('run_change', 'Smaller 3 hidden layer NN, no DO' + str(params))

        torch.backends.cudnn.benchmark = True
        print('total epochs=', len(range(start_epoch, start_epoch+params['epochs'])))

        # with torch.autograd.detect_anomaly():  # use this to detect bugs while training
        for param_group in optimizer.param_groups:
            print('lr-check', param_group['lr'])
        for epoch in range(start_epoch, start_epoch+params['epochs']):  # loop over the dataset multiple times
            running_loss = 0.0
            stime = time.time()

            for i, data in enumerate(train_dl, 0):
                iter_count += 1
                # get the inputs; data is a list of [inputs, dist_true]
                model.train()
                inputs, dist_true = data[0], data[1]
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = loss_fn(outputs, dist_true)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                last_loss = loss.item()

                for param_group in optimizer.param_groups:
                    curr_lr = param_group['lr']
                writer.add_scalar('monitor/lr-iter', curr_lr, iter_count-1)

                if not isinstance(lr_sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_sched.step()

            val_loss = evaluate(model, val_dl)
            if isinstance(lr_sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_sched.step(val_loss)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                patience_counter = 0
                best_model = copy.deepcopy(model)
                print(epoch,"> Best val_loss model saved:", round(val_loss, 4))
            else:
                patience_counter += 1
            train_loss = running_loss/len(train_dl)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            writer.add_scalar('loss/train', train_loss, epoch)
            writer.add_scalar('loss/val', val_loss, epoch)
            for param_group in optimizer.param_groups:
                curr_lr = param_group['lr']
            writer.add_scalar('monitor/lr-epoch', curr_lr, epoch)
            if patience_counter > early_stop_patience:
                print("Early stopping at epoch {}. current val_loss {}".format(epoch, val_loss))
                break
               
            """
            if epoch % 10 == 0:
                torch.save(best_model.state_dict(), os.path.join(checkpoint_path, 'model_cp.pt'))
                torch.save(optimizer.state_dict(), checkpoint_path+'/optim_cp.pt')
                torch.save(lr_sched.state_dict(), checkpoint_path+'/sched_cp.pt')
                writer.add_text('checkpoint saved', 'at epoch='+str(epoch))
            """
               
            print("epoch:{} -> train_loss={},val_loss={} - {}".format(epoch, round(train_loss, 5),round(val_loss, 5), seconds_to_minutes(time.time()-stime)))
             

        print('Finished Training')
        ts = str(time.time())
        best_model_path = os.path.join(checkpoint_path,'model_'+ts+'.pt')
        opt_save_path = os.path.join(checkpoint_path, 'optim_'+ts+'.pt')
        sched_save_path = os.path.join(checkpoint_path, 'sched_'+ts+'.pt')
        state_save_path = os.path.join(checkpoint_path, 'state_'+ts+'.pt')
        state = {'epoch': epoch+1,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'last_train_loss': train_losses[-1],
                'last_val_loss': val_losses[-1],
                'total_iters': iter_count
                }

        save_checkpoint(state, state_save_path)
        # sometimes loading from state dict is not wokring, so...
        torch.save(best_model.state_dict(), best_model_path)
        torch.save(optimizer.state_dict(), opt_save_path)
        torch.save(lr_sched.state_dict(), sched_save_path)

        def test(model, dl):
            model.eval()
            final_loss = 0.0
            count = 0
            y_hat = []
            with torch.no_grad():
                for data_cv in dl:
                    inputs, dist_true = data_cv[0], data_cv[1]
                    count += len(inputs)
                    outputs = model(inputs)
                    y_hat.extend(outputs.tolist())
                    loss = loss_fn(outputs, dist_true)
                    final_loss += loss.item()
            return final_loss/len(dl), y_hat

        model.load_state_dict(torch.load(best_model_path))
        test_loss, y_hat = test(model, test_dl)
        print(test_loss)
        writer.add_text('test-loss', str(test_loss))
        try:
            if scaler:
                y_hat = scaler.inverse_transform(y_hat)
                self.y_test = scaler.inverse_transform(self.y_test)
        except:
            pass

        acc_score = accuracy_score(self.y_test[:len(y_hat)], np.round(y_hat))

        writer.add_text('Accuracy=', str(acc_score))
        print(str(accuracy_score(self.y_test[:len(y_hat)], np.round(y_hat))))

        y_hat_ = np.array(y_hat).squeeze()
        self.y_test_ = self.y_test[:len(y_hat)]
        print(len(self.y_test), len(y_hat))
        dist_accuracies = []
        dist_mae = []
        dist_mre = []
        dist_mse = []
        dist_counts = []
        for i in range(self.max_dist + 1):
            mask = self.y_test_==i
            dist_values = self.y_test_[mask]
            dist_preds = np.round(y_hat_[mask])
            if len(dist_values) != len(dist_preds):
                print("ERROR: len(dist_values) != len(dist_preds) => {} != {}".format(len(dist_values), len(dist_preds)))
            elif len(dist_values) < 1 and i != 0:
                dist_accuracies.append(np.nan)
                dist_mae.append(np.nan)
                dist_mse.append(np.nan)
                dist_mre.append(np.nan)
                dist_counts.append(len(dist_values))
                continue
            elif len(dist_values) < 1 and i == 0:
                continue
            dist_accuracies.append(np.sum(dist_values == dist_preds)*100/len(dist_values))
            dist_mae.append(mean_absolute_error(dist_values, dist_preds))
            dist_mse.append(mean_squared_error(dist_values, dist_preds))
            dist_mre.append(mean_absolute_percentage_error(dist_values, dist_preds))
            dist_counts.append(len(dist_values))

        """
        fig = plt.figure(figsize=(10,7))
        plt.subplot(2,1,1)
        plt.bar(range(18), dist_accuracies)
        for index, value in enumerate(dist_accuracies):
            plt.text(index+0.03, value, str(np.round(value, 2))+'%')
        plt.title('distance-wise accuracy')
        plt.xlabel('distance values')
        plt.ylabel('accuracy')
        plt.subplot(2,1,2)
        plt.bar(range(18), dist_counts)
        for index, value in enumerate(dist_counts):
            plt.text(index+0.03, value, str(value))
        plt.title('distance-wise count')
        plt.xlabel('distance values')
        plt.ylabel('counts')
        fig.tight_layout(pad=3.0)
        im_path = os.path.join(self.path, "accuracy scores.png")
        self.ensure_folders_exist(im_path)
        fig.savefig(im_path)
        writer.add_figure('test/results', fig)
        """
        
        writer.add_text('class avg accuracy', str(np.mean(dist_accuracies)))
        print('class avg accuracy', np.mean(dist_accuracies))

        mse = mean_squared_error(np.array(y_hat).squeeze(), self.y_test[:len(y_hat)])

        writer.add_text('MSE', str(mse))
        print('MSE', mse)

        mae = mean_absolute_error(np.array(y_hat).squeeze(), self.y_test[:len(y_hat)])

        writer.add_text('MAE', str(mae))
        print('MAE', mae)
        
        mre = mean_absolute_percentage_error(np.array(y_hat).squeeze(), self.y_test[:len(y_hat)])

        writer.add_text('MRE', str(mre))
        print('MRE', mre)

        self.scores["nn"] = {"acc": acc_score, "mse": mse, "mae": mae, "mre": mre, "lr_arr": lr_arr[:len(lrs)], "lr_losses": losses, "train_losses": train_losses, "val_losses": val_losses, "dist_accuracies": dist_accuracies, "dist_mae": dist_mae, "dist_mre": dist_mre, "dist_mse": dist_mse, "dist_counts": dist_counts}
