def create_bow(ids, dictionary_size):
    bow = [0] * dictionary_size
    for id in ids:
        bow[id] += 1
    return bow


dict = ["czarna", "czerwona", "herbata", "najwięcej",
        "niż", "teina", "więcej", "zawierać"]
ids1 = [3, 5, 7, 2, 0]  # Najwięcej teiny zawiera herbata czarna.
ids2 = [0, 2, 7, 6, 5, 4, 2, 1]  # Czarna herbata zawiera więcej teiny niż herbata czerwona.
ids3 = ids1 + ids2  # Najwięcej teiny zawiera herbata czarna. Czarna herbata zawiera więcej teiny niż herbata czerwona.

bow1 = create_bow(ids1, len(dict))
bow2 = create_bow(ids2, len(dict))
bow3 = create_bow(ids3, len(dict))

print(bow1)
# [1, 0, 1, 1, 0, 1, 0, 1]
print(bow2)
# [1, 0, 1, 1, 0, 1, 0, 1]
print(bow3)
# [2, 1, 3, 1, 1, 2, 1, 2]
