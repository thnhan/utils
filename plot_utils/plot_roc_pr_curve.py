"""
@author: thnhan
ref: https://stackoverflow.com/questions/29656550/how-to-plot-pr-curve-over-10-folds-of-cross-validation-in-scikit-learn
"""

import numpy as np
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    auc
)


def plot_folds(plt, arr_true_y: list, arr_prob_y: list, figsize='big'):
    # init
    if figsize == 'big':
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))  # big
    elif figsize == 'small':
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))  # small
    else:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)  # custom

    ax_roc = ax[0]
    ax_rpc = ax[1]

    n_samples = 0
    for i, y_test in enumerate(arr_true_y):
        n_samples += y_test.shape[0]
    mean_fpr = np.linspace(0, 1, n_samples)
    roc_aucs = []
    tprs = []

    # get fpr, tpr scores
    for i, (y_test, y_prob) in enumerate(zip(arr_true_y, arr_prob_y)):
        # # print(y_test)

        # if np.ndim(y_prob) > 1:
        #     y_prob = y_prob[:, 1]  # only use prob of class 1

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        # plot ROC curve
        ax_roc.plot(fpr, tpr, lw=1, alpha=0.5, label='Fold {:d} (AUC = {:0.2%})'.format(i + 1, roc_auc))

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        roc_aucs.append(roc_auc)

    # ax_roc.plot([0, 1], [0, 1], lw=2, alpha=.8, linestyle='--', color='r', label='Chance')
    ax_roc.plot([0, 1], [0, 1], lw=1, alpha=.8, linestyle='--', color='r',
                # label='Chance'
                )

    # Ve ROC mean
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_roc_auc = auc(mean_fpr, mean_tpr)
    std_roc_auc = np.std(roc_aucs)
    ax_roc.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean (AUC = {:0.2%} $\pm$ {:0.2%})'.format(mean_roc_auc, std_roc_auc),
                lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax_roc.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey',
                        alpha=.2)  # ,label_new=r'$\pm$ 1 std. dev.')

    # Dat ten 
    ax_roc.set(
        xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title="Receiver Operating Characteristic Curve"
    )
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend(loc="lower right")

    ####################################################################################
    # mean_recall = np.linspace(0, 1, n_samples)
    # pres = []; rpc_aucs = []
    # get precision, recall scores
    for i, (y_test, y_prob) in enumerate(zip(arr_true_y, arr_prob_y)):
        # if np.ndim(y_prob) > 1:
        #     y_prob = y_prob[:, 1]  # only use prob of class 1
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        average_precision = average_precision_score(y_test,
                                                    y_prob)  # Quay lại sử dụng lệnh này nếu có sai sót
        # plot precision recall curve

        ax_rpc.plot(
            recall, precision,
            lw=1, alpha=0.5, label='Fold {:2d} (AP = {:0.2%})'.format(i + 1, average_precision)
        )

        # interp_precision = np.interp(mean_recall, recall, precision)
        # # interp_precision[0] = 0.0
        # pres.append(interp_precision)
        # rpc_aucs.append(rpc_auc)

    y_tests = np.array([])
    for y_test in arr_true_y:
        y_tests = np.hstack((y_tests, y_test.ravel()))

    # # === Plot Noskill line
    # no_skill = len(y_tests[y_tests == 1]) / y_tests.shape[0]
    # ax_rpc.plot(
    #     [0, 1], [no_skill, no_skill],
    #     linestyle='--', lw=2, color='r', label='Chance'
    # )

    # Ve duong mean
    all_y_test = np.concatenate(arr_true_y)
    all_y_prob = np.concatenate(arr_prob_y)
    precision, recall, _ = precision_recall_curve(all_y_test, all_y_prob)

    ax_rpc.plot(
        recall, precision, color='b',
        label=r'Overall (AP = {:0.2%})'.format(average_precision_score(all_y_test, all_y_prob)),
        lw=2, alpha=.8
    )

    # Dat ten
    ax_rpc.set_title(' Precision-Recall Curve')
    ax_rpc.set_xlabel('Recall')
    ax_rpc.set_ylabel('Precision')
    ax_rpc.legend(loc="lower left")


def plot_mean_kfolds(plt, arr_true_y: list, arr_prob_y: list, figsize='big'):
    # === init
    if figsize == 'big':
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))  # big
    elif figsize == 'small':
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))  # small
    else:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)  # custom

    ax_roc = ax[0]  # cho ROC curve
    ax_prc = ax[1]  # cho PR curve

    # ###########################################################
    # Ve ROC
    # ###########################################################
    mean_fpr, mean_tpr, mean_AUC_ROC, std_AUC_ROC = mean_std_AUC_ROC(arr_true_y, arr_prob_y)

    ax_roc.plot(mean_fpr, mean_tpr, color='blue',
                label=r'Mean (AUC={:0.2%}$\pm${:0.2%})'.format(mean_AUC_ROC, std_AUC_ROC),
                lw=2, alpha=.8)

    # == Ve no_skill
    # ax_roc.plot([0, 1], [0, 1], lw=2, alpha=.8, linestyle='--', color='r', label='Chance')
    ax_roc.plot([0, 1], [0, 1], lw=0.75, alpha=.8, linestyle='--', color='darkred')

    # Dat ten
    ax_roc.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    ax_roc.set_title(
        label="Receiver Operating Characteristic Curve",
        # fontname="Times New Roman",
        # size=28,
        fontweight="bold"
    )
    ax_roc.set_xlabel('False Positive Rate',
                      # fontname="Times New Roman",
                      # size=28,
                      fontweight="bold"
                      )
    ax_roc.set_ylabel('True Positive Rate',
                      # fontname="Times New Roman",
                      # size=28,
                      fontweight="bold"
                      )
    ax_roc.legend(loc="lower right")

    # ###########################################################
    # Ve PR Curve
    # ###########################################################
    all_y_test = np.concatenate(arr_true_y)
    all_y_prob = np.concatenate(arr_prob_y)
    precision, recall, _ = precision_recall_curve(all_y_test, all_y_prob)

    ax_prc.plot(
        recall, precision, color='b',
        label=r'AP={:0.2%}'.format(average_precision_score(all_y_test, all_y_prob)),
        lw=2, alpha=.8
    )

    # === Dat ten
    ax_prc.set_title('Precision-Recall Curve',
                     # fontname="Times New Roman",
                     # size=28,
                     fontweight="bold"
                     )
    ax_prc.set_xlabel('Recall',
                      # fontname="Times New Roman",
                      # size=28,
                      fontweight="bold"
                      )
    ax_prc.set_ylabel('Precision',
                      # fontname="Times New Roman",
                      # size=28,
                      fontweight="bold"
                      )
    ax_prc.legend(loc="lower left")


def mean_std_AUC_ROC(arr_true_y: list, arr_prob_y: list):
    """ @thnhan
        Lay gia tri mean va std tren K-fold cua duong Receiver Operating Characteristic Curve (ROC).
        """

    n_samples = 0
    for i, y_test in enumerate(arr_true_y):
        n_samples += y_test.shape[0]
    mean_fpr = np.linspace(0, 1, n_samples)

    roc_aucs = []
    tprs = []

    # === get fpr, tpr scores
    for i, (y_test, y_prob) in enumerate(zip(arr_true_y, arr_prob_y)):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        roc_aucs.append(roc_auc)

    # === Get mean value, and std value
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_AUC_ROC = auc(mean_fpr, mean_tpr)
    std_AUC_ROC = np.std(roc_aucs)

    return mean_fpr, mean_tpr, mean_AUC_ROC, std_AUC_ROC


def ave_std_AUC_PRC(arr_true_y: list, arr_prob_y: list):
    """ @thnhan
        Lay gia tri mean va std tren K-fold cua duong Precision-Recall Curve (PRC).
        """
    y_tests = np.array([])
    for y_test in arr_true_y:
        y_tests = np.hstack((y_tests, y_test.ravel()))

    # === Gop chung K-Fold thanh mot va tinh trung binh
    all_y_test = np.concatenate(arr_true_y)
    all_y_prob = np.concatenate(arr_prob_y)
    ave_precision, ave_recall, _ = precision_recall_curve(all_y_test, all_y_prob)

    return all_y_test, all_y_prob, ave_precision, ave_recall


def plot_ROC_PRC_methods(plt,
                         methods_result: dict,
                         figsize='big',
                         cmap_list=None):
    # === init
    if figsize == 'big':
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))  # big
    elif figsize == 'small':
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))  # small
    else:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)  # custom

    # plt.set_cmap("Dark2")

    ax_roc = ax[0]  # cho ROC curve
    ax_prc = ax[1]  # cho PR curve

    # init
    methods_name = list(methods_result.keys())
    methods_label_pred = list(methods_result.values())
    # arr_y_test = [methods_label_pred[i][0] for i in range(len(methods_result))]
    # arr_y_prob = [methods_label_pred[i][1] for i in range(len(methods_result))]

    arr_y_test = [methods_label_pred[i][0] for i in range(len(methods_result))]
    arr_y_prob = [methods_label_pred[i][1] for i in range(len(methods_result))]

    # get fpr, tpr scores
    for i, (name_method, y_test, y_prob) in enumerate(zip(methods_name, arr_y_test, arr_y_prob)):
        # ###########################################################
        # Ve ROC
        # ###########################################################
        mean_fpr, mean_tpr, mean_AUC_ROC, std_AUC_ROC = mean_std_AUC_ROC(y_test, y_prob)

        if type(cmap_list) == 'list':
            color_ix = cmap_list[i]
        else:
            color_ix = cmap_list(i)

        if name_method == 'DeepCF-PPI*':
            ax_roc.plot(mean_fpr, mean_tpr,
                        label=r'{} ({:0.3}$\pm${:0.2%})'.format(name_method, mean_AUC_ROC, std_AUC_ROC),
                        lw=1,
                        # alpha=1.,
                        # color=cmap_list(7-i)
                        color=color_ix
                        )
        else:
            ax_roc.plot(mean_fpr, mean_tpr,
                        label=r'{} ({:0.3}$\pm${:0.2%})'.format(name_method, mean_AUC_ROC, std_AUC_ROC),
                        lw=1,
                        # alpha=1.,
                        # color=cmap_list(7-i)
                        color=color_ix
                        )

    # == Ve no_skill
    # ax_roc.plot([0, 1], [0, 1], lw=2, alpha=.8, linestyle='--', color='r', label='Chance')
    ax_roc.plot([0, 1], [0, 1], lw=0.75, alpha=.8, linestyle='--')

    # Dat ten
    ax_roc.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    ax_roc.set_title(
        label="AUROC",  # "Receiver Operating Characteristic Curve",
        # fontname="Times New Roman",
        # size=28,
        # fontweight="bold"
    )
    ax_roc.set_xlabel('False Positive Rate',
                      # fontname="Times New Roman",
                      # size=28,
                      # fontweight="bold"
                      )
    ax_roc.set_ylabel('True Positive Rate',
                      # fontname="Times New Roman",
                      # size=28,
                      # fontweight="bold"
                      )
    ax_roc.legend(loc="lower right", fontsize='small',
                  # prop=dict(weight='bold'),
                  )

    for i, (name_method, y_test, y_prob) in enumerate(zip(methods_name, arr_y_test, arr_y_prob)):
        # ###########################################################
        # Ve PR Curve
        # ###########################################################
        _y_test = np.concatenate(y_test)
        _y_prob = np.concatenate(y_prob)

        if name_method == 'Ada':
            # _y_test = np.concatenate((y_test[0], y_test[1], y_test[2], y_test[3], y_test[4]))
            # _y_prob = np.concatenate((y_prob[0], y_test[1], y_prob[2], y_prob[3], y_prob[4]))
            _y_test = np.concatenate((y_test[0]))
            _y_prob = np.concatenate((y_prob[0]))

        precision, recall, thres = precision_recall_curve(_y_test, _y_prob)

        # if name_method == 'Ada':
        #     # print(list(precision))
        #     for ii, pre in enumerate(precision):
        #         if pre < 0.52:
        #             print(thres[ii])

        if type(cmap_list) == 'list':
            color_ix = cmap_list[i]
        else:
            color_ix = cmap_list(i)

        ax_prc.plot(
            recall, precision,
            label=r'{} ({:0.3f})'.format(name_method, average_precision_score(_y_test, _y_prob)),
            lw=1,
            # alpha=1.,
            # color=cmap_list(7-i)
            color=color_ix
        )

    # === Dat ten
    ax_prc.set_title('AUPRC',  # 'Precision-Recall Curve',
                     # fontname="Times New Roman",
                     # size=28,
                     # fontweight="bold"
                     )
    ax_prc.set_xlabel('Recall',
                      # fontname="Times New Roman",
                      # size=28,
                      # fontweight="bold"
                      )
    ax_prc.set_ylabel('Precision',
                      # fontname="Times New Roman",
                      # size=28,
                      # fontweight="bold"
                      )
    # legend_properties = {'weight': 'bold', 'loc':"lower left"}
    ax_prc.legend(
        # prop=dict(weight='bold'),
        loc='lower left'
        , fontsize='small',
        # fontname="Consolas",
    )


def plot_methods(methods_name_and_y_prob: dict, save=None):
    """
    @author: thnhan

    Parameters:
    ==========================

    `methods_result`: `dict('method name', [y_true, y_prob])`
    """
    import matplotlib.pyplot as plt

    # init
    name_methods = list(methods_name_and_y_prob.keys())
    tam = list(methods_name_and_y_prob.values())
    # print(methods_result)
    arr_y_test = [tam[i][0] for i in range(len(methods_name_and_y_prob))]
    arr_y_prob = [tam[i][1] for i in range(len(methods_name_and_y_prob))]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    ax_roc = ax[0]
    ax_rpc = ax[1]

    # get fpr, tpr scores
    for i, (name_method, y_test, y_prob) in enumerate(zip(name_methods, arr_y_test, arr_y_prob)):
        # if np.ndim(y_prob) > 1:
        #     y_prob = y_prob[:, 1]  # only use prob of class 1
        fpr, tpr, _ = roc_curve(y_test.ravel(), y_prob.ravel())
        roc_auc = auc(fpr, tpr)
        # plot ROC curve
        ax_roc.plot(
            fpr, tpr,
            lw=1.5, alpha=1,
            label=name_method + ' (AUC = %0.4f)' % roc_auc
        )

        # interp_tpr = np.interp(mean_fpr, fpr, tpr)
        # interp_tpr[0] = 0.0
        # tprs.append(interp_tpr)
        # roc_aucs.append(roc_auc)

    ax_roc.plot([0, 1], [0, 1], lw=2, alpha=.8, linestyle='--', color='r')  # , label_new='Chance')

    # Dat ten 
    ax_roc.set(
        xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
        title="Receiver operating characteristic curve"
    )
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend(loc="lower right")

    ####################################################################################
    # mean_recall = np.linspace(0, 1, n_samples)
    # pres = []; rpc_aucs = []
    # get precision, recall scores
    for i, (name_method, y_test, y_prob) in enumerate(zip(name_methods, arr_y_test, arr_y_prob)):
        # if np.ndim(y_prob) > 1:
        #     y_prob = y_prob[:, 1]  # only use prob of class 1
        precision, recall, _ = precision_recall_curve(y_test.ravel(), y_prob.ravel())
        average_precision = average_precision_score(y_test,
                                                    y_prob)  # Quay lại sử dụng lệnh này nếu có sai sót
        # plot precision recall curve

        ax_rpc.plot(
            recall, precision,
            lw=1.5, alpha=1,
            label=name_method + ' (AUPR = %.4f)' % average_precision
        )

        # interp_precision = np.interp(mean_recall, recall, precision)
        # # interp_precision[0] = 0.0
        # pres.append(interp_precision)
        # rpc_aucs.append(rpc_auc)

    """CŨ"""
    # y_tests = np.array([])
    # for y_test in arr_true_y:
    #     y_tests = np.hstack((y_tests, y_test.ravel()))
    """MỚI"""
    y_tests = np.array(arr_y_test).ravel()

    # no_skill = len(y_tests[y_tests == 1]) / y_tests.shape[0]
    # ax_rpc.plot(
    #     [0, 1], [no_skill, no_skill],
    #     linestyle='--', lw=2, color='r', label_new='Chance'
    # )

    # Dat ten
    ax_rpc.set_title('Precision-Recall curve')
    ax_rpc.set_xlabel('Recall')
    ax_rpc.set_ylabel('Precision')
    ax_rpc.legend(loc="lower left")

    if save is not None:
        fig.savefig(save + '.eps', format='eps',
                    transparent=True,
                    bbox_inches='tight')
        fig.savefig(save + '.png', format='png',
                    transparent=True,
                    bbox_inches='tight')
    plt.show()
