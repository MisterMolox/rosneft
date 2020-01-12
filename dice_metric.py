def dice(y_true, y_pred):                                  # Метрика сходства двух изображений
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

def dice_border(y_true, y_pred):
    n, m = y_true.shape
    y_pred_border = np.zeros((4, m))
    for i in range(m):
        j_up = n - 1
        j_down = 0
        for j in range(n):
            if 1 == y_true[j, i]:
                if j_up > j:
                    j_up = j
            if 0 == y_true[j, i]:
                if j_down < j:
                    j_down = j
        assert(j_down - j_up >= 3)
        y_pred_border[0, i] = y_pred[j_up, i]
        y_pred_border[1, i] = y_pred[j_up + 1, i]
        y_pred_border[2, i] = y_pred[j_down - 1, i]
        y_pred_border[3, i] = y_pred[j_down, i]
    return dice(np.ones((4, m)), y_pred_border)

def dice_coef(y_true, y_pred):                            
    return (dice(y_true, y_pred) + dice_border(y_true, y_pred)) /2 

def dice_coef_loss(y_true, y_pred):                        # Функция потерь для минимизации
    return -dice_coef(y_true, y_pred)
