import turtle

def caise5():
    """绘制彩色五边形"""
    t = turtle.Pen()
    turtle.bgcolor("black")
    sides = 6
    colors = ["red", "yellow", "green", "blue", "orange", "purple"]
    for x in range(360):
        t.pencolor(colors[x % sides])
        t.forward(x*3/sides+x)
        t.left(360/sides+1)
        t.width(x*sides/200)

def draw_tree():
    def draw_brach(brach_length):
        if brach_length > 5:
            if brach_length < 40:
                turtle.color('green')

            else:
                turtle.color('red')

            # 绘制右侧的树枝
            turtle.forward(brach_length)
            print('向前',brach_length)
            turtle.right(25)
            print('右转20')
            draw_brach(brach_length-15)
            # 绘制左侧的树枝
            turtle.left(50)
            print('左转40')
            draw_brach(brach_length-15)

            if brach_length < 40:
                turtle.color('green')

            else:
                turtle.color('red')

            # 返回之前的树枝上
            turtle.right(25)
            print('右转20')
            turtle.backward(brach_length)
            print('返回',brach_length)
    def __main():
        turtle.left(90)
        turtle.penup()
        turtle.backward(150)
        turtle.pendown()
        turtle.color('red')

        draw_brach(100)

        turtle.exitonclick()

    __main()

import turtle

def koch_roc(N):
    """N阶科赫曲线 N=2-10最好
    来自：https://blog.csdn.net/qq_35527032/article/details/81023121
    """
    def koch(size, n):
        if n == 0:
            turtle.fd(size)
        else:
            for angle in [0, 60, -120, 60]:
                turtle.left(angle)
                koch(size/3, n-1)

    def __main():
        turtle.setup(1200, 760, 0, 0)
        turtle.speed(200)
        turtle.penup()
        turtle.goto(-300, 200)
        turtle.pendown()
        turtle.pensize(2)
        level = N
        koch(600, level)  # 3阶科赫曲线
        turtle.right(120)
        koch(600, level)
        turtle.right(120)
        koch(600, level)
        turtle.hideturtle()
        turtle.done()
    __main()


koch_roc(3)
