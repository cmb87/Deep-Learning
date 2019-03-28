from src.neuralnetwork import *
from sklearn.datasets import make_blobs


if __name__ == "__main__":

    features, labels = make_blobs(n_samples=50, n_features=2, centers=2, random_state=75)

    x = np.linspace(0, 11, 10)
    y = -x +5


    #print(np.array([1, 1]).dot(np.array([8, 10])) - 5) # ==> 13 > 0 ==> belongs to upper class
    #print(np.array([1, 1]).dot(np.array([4, 0])) - 5)  # ==> -1 < 0 ==> belongs to lower class

    #plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='coolwarm')
    #plt.plot(x,y)
    #plt.show()


    g = Graph()
    g.set_as_default()

    x = Placeholder()
    w = Variable([1,1])
    b = Variable(-5)

    z = add(matmul(w,x),b)
    a = sigmoid(z)

    sess = Session()
    res = sess.run(operation=a, feed_dict={x:[8,10]})

    print(res)