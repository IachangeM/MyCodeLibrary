/*
from <C和指针>
@author:geekac
*/


//Linux源码中关于min的宏定义：【注意  typeof是Linux C中定义的函数！】
#define min(x, y) ({     \
typeof(x) _x = (x);    \
typeof(y) _y = (y);    \
(void) (&_x == &_y);   \
_x < _y ? _x : _y; })

/*
1  其实关于min的宏，更好的做法是再加个const

2  (void) (&_x == &_y); 中的void，表示将表达式(&_x == &_y); 
   所得到的结果（此处肯定是逻辑上的假，值为0）忽略掉。如果不加void，则会提示你这行代码是无意义的，没人用到。

3  关于min的宏定义，为何这么复杂，而不是用简单的#define min(x,y) ((x) < (y) ? x : y)
因为，如果如此定义，那么对于一些特殊的值传入此宏之后，就会产生一些副作用，产生的结果，就不是我们想要的了，比如：
min(++a,++b) ==> ((++a)<(++b))?(++a) : (++b) 




 */


#define Cmin(x, y) ({ \
const typeof(x) _x = (x); \
const typeof(y) _y = (y); \
(void) (&_x == &_y); \
_x < _y ? _x : _y; })



int main(){
    /*  一、 C、C++代码换行问题      */
    /*
        1. 预处理一行写不下： 
            把一个预处理指示写成多行要用“\"续行，因为根据定义，一条预处理指示只能由一个逻辑代码行组成。 
        2. 字符串常量跨行 
            在行尾使用“\"，然后回车换行，就可以吧字符串常量跨行书写，注意下一行顶格写 
        3. 正常程序一行写不下： 
        　　把C代码写成多行则不必使用续行符，因为换行在C代码中只不过是一种空白符，在做语法解析（语法分析）时所有空白符都被丢弃了。 
    */
    char letters[] = {"abcdefghijklmnopqrstuvwxyz\
ABCDEFGHIJKLMNOPQRSTUVWXYZ"};   // 注意顶格写！！ 不然会中间有多余的空格



    /*  二、 #define宏定义      */
    /*
        1、明确记住一点：#define是宏替换，在代码编译之前进行源码字符串替换
        2、两种宏：无参宏定义和带参宏定义
            (1) 无参宏定义： #define 标识符 字符串
                需要注意(字符串) 使用括号括起来，以免发生错误。
            
            (2) 带参宏定义： #define 宏名(形参表) 字符串
            同样注意括号的问题！！ #define CalcInterest(x, y) ( (x) * (y) )
        
        3、标识符/宏名一般使用大写！
        
        
       【注意】  如果宏定义较长 则使用 \ 进行换行书写。


        4、#算符可产生一个C语言格式的字符串。
        例如：#define string（x）#x
        此时NSLog(@"%s", string(testing)); // 输出为testing,
        注意,C语言格式的字符串需要用%s,不能用%@
        
        5、##算符用来连接两段字符串 【有待整理】
        #define LINK3(x,y,z) x##y##z
        LINK3("C", "+", "+") 连接后的结果是 "C++"
        LINK3(3,5,0)  ##将数字也视为字符串 因此连接后的结果是"350"
    

    //  其他参考链接：https://www.cnblogs.com/southcyy/p/10155049.html
    */




    int x = 2;
    char y = 3;
    int m;
    m = min(x,y);


    return 0;
}

