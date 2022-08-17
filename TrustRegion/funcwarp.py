def call(x,y):
    calltime = 10
    def sub(x):
        return x + calltime
    calltime = 114514
    print(sub(1024+x+y))
if __name__ == "__main__":
    call(12,11)
