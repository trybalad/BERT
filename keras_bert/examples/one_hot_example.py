from tensorflow.keras.utils import to_categorical

dict = ["czarna", "herbata", "najwięcej", "teina", "zawierać"]
ids = [2, 3, 4, 1, 0]  # Najwięcej teiny zawiera herbata czarna.

one_hot = to_categorical(ids, len(dict))

print(one_hot)

"""
[[0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]
 [0. 1. 0. 0. 0.]
 [1. 0. 0. 0. 0.]]
"""
