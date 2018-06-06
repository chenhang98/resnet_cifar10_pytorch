from matplotlib import pyplot as plt 
import csv

def watch(fn):
    reader = csv.reader(open(fn))
    iters, acc, loss = [], [], []
    val_acc, val_loss = [], []

    names = next(reader)
    for row in reader:        
        loss.append(float(row[1]))
        val_loss.append(float(row[2]))
        
        acc.append(float(row[3]))
        val_acc.append(float(row[4]))

        
    print("last acc/val_acc: ", acc[-1], val_acc[-1])  
    print("max acc/val_acc: ", max(acc), max(val_acc))

    plt.subplot(211)
    plt.plot(acc, 'b', label = "acc")    
    plt.plot(val_acc, 'r', label = "val_acc")
    plt.legend()

    plt.subplot(212)
    plt.plot(loss[1:], 'b', label = "loss")
    plt.plot(val_loss[1:], 'r', label = "val_loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
	watch("log.csv")