import matplotlib.pyplot as plt
import pickle

acc = pickle.load(open('acc.p', 'rb'))
  
plt.plot(acc, '' )
plt.title('NuPIC learning with MNIST')
plt.xlabel('i')
plt.ylabel('accuracy')
plt.show()