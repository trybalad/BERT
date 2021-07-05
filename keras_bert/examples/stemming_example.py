from nltk.stem import LancasterStemmer
from nltk.stem import PorterStemmer

words = ["dogs", "dancing", "running", "grew",
         "disobedience", "troublesome", "easiest",
         "harder", "rational", "irrational",
         "kilogram", "nanometer"]

porter = PorterStemmer()
lancaster = LancasterStemmer(strip_prefix_flag=True)

print([porter.stem(word) for word in words])
# ['dog', 'danc', 'run', 'grew',
# 'disobedi', 'troublesom', 'easiest',
# 'harder', 'ration', 'irrat',
# 'kilogram', 'nanomet']
print([lancaster.stem(word) for word in words])
# ['dog', 'dant', 'run', 'grew',
# 'disobedy', 'troublesom', 'easiest',
# 'hard', 'rat', 'ir',
# 'gram', 'met']
