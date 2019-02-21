
#include<stdio.h>
#include<stdlib.h>
#include<queue>
#define N 11
#define NoChild 10
using namespace std;

地址：https://pintia.cn/problem-sets/16/problems/666

/*这道题还是使用结构数组更合适，因为找到的rootId是数字
struct Node{
    int val;
    struct Node* left;
    struct Node* right;
};
typedef struct Node *ptrNode;
*/

struct Node{
    int left;
    int right;
};
typedef struct Node Node;
Node Tree[N];
bool findRootAssist[N];

int charToChild(char c){
    if(c=='-') return NoChild;
    return c-'0';
}


int main(){
    for(int i=0; i<N; i++){
        findRootAssist[i] = true;
        Tree[i].left  = NoChild;
        Tree[i].right = NoChild;
    }
    int n;scanf("%d", &n);//一共n个节点

    char left_, right_;
    int left, right;
    for(int i=0;i<n;i++){
        getchar();
        scanf("%c %c", &left_, &right_);
        left = charToChild(left_);
        right = charToChild(right_);

        Tree[i].left = left;
        Tree[i].right = right;

        findRootAssist[left] = false;
        findRootAssist[right] = false;
    }


    int rootId = 0;
    for(;rootId<n && !findRootAssist[rootId];rootId++);

    //下面开始进行层序遍历。需要借助队列
    queue<int> Q;
    Q.push(rootId);
    int crtNode;
    while(!Q.empty()){
        crtNode = Q.front();
        if(Tree[crtNode].left==NoChild && Tree[crtNode].right==NoChild)
            printf("%d ", crtNode);
        //左孩子入队



    }

    getchar();
    return 0;
}






