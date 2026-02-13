# 1、两数之和

![](C:\Users\Jackson Zhang\AppData\Roaming\Typora\typora-user-images\image-20260207150938702.png)

```go
func twoSum(nums []int, target int) []int {
    hasMap := make(map[int]int)
    for i,v := range nums{
        if p,ok := hasMap[target-v];ok{
            return []int{p,i}
        }else{
            hasMap[v] = i
        }
    }
    return nil
}
```

# 2、字母异位词分组

![image-20260207151051716](C:\Users\Jackson Zhang\AppData\Roaming\Typora\typora-user-images\image-20260207151051716.png)

```go
//字母异位词特点：排序得到的字符串相同、每个字母出现的次数相同
func groupAnagrams(strs []string) [][]string {
    hasMap := make(map[string][]string)
    for _,s := range strs{
        strB := []byte(s)
        sort.Slice(strB,func(i,j int) bool {return strB[i] < strB[j]})
        sortedStr := string(strB)
        hasMap[sortedStr] = append(hasMap[sortedStr],s)
    }
    res := make([][]string,0,len(hasMap))
    for _,v := range hasMap{
        res = append(res,v)
    }
    return res
}
```

# 3.最长连续序列

![image-20260207153411117](C:\Users\Jackson Zhang\AppData\Roaming\Typora\typora-user-images\image-20260207153411117.png)

```go
func longestConsecutive(nums []int) int {
    numsMap := make(map[int]bool)
    ans := 0
    for _,v := range  nums{
        numsMap[v] = true
    }
    for x:= range numsMap{
        if numsMap[x-1] {
            continue
        }
        y := x+1
        for numsMap[y]{
            y++
        }

        ans = max(ans,y-x)
    }
    return ans
}
```

# 4.移动零

![image-20260207160803911](C:\Users\Jackson Zhang\AppData\Roaming\Typora\typora-user-images\image-20260207160803911.png)

```go
func moveZeroes(nums []int)  {
    stackSize :=0
    for _,x :=range nums{
        if x != 0{
            nums[stackSize] = x
            stackSize++
        }
    }
    clear(nums[stackSize:])
}
```

# 5、盛最多的水

![image-20260208133027647](C:\Users\Jackson Zhang\AppData\Roaming\Typora\typora-user-images\image-20260208133027647.png)

```go
func maxArea(height []int) int {
    left := 0
    right := len(height)-1
    res := 0
    for left < right{
        area := (right-left) * min(height[left],height[right])
        if height[left] < height[right]{
            left++
        }else{
            right--
        }
        res = max(res,area)
    }
    return res
}
```

# 6、三数之和

![image-20260208141008243](C:\Users\Jackson Zhang\AppData\Roaming\Typora\typora-user-images\image-20260208141008243.png)

```go
//三指针，i,left,right
func threeSum(nums []int) [][]int {
    res := make([][]int,0)
    slices.Sort(nums)
    for i:=0;i<len(nums);i++{
        if nums[i] > 0{
            break
        }
        if i>0 && nums[i] == nums[i-1]{
            continue
        }
        left := i+1
        right := len(nums)-1
        for left<right{
            if nums[i] + nums[left] +nums[right] > 0{
                right--
            }else if nums[i] + nums[left] + nums[right] < 0{
                left++
            }else{
                res = append(res,[]int{nums[i],nums[left],nums[right]})
                for left < right && nums[left] == nums[left+1]{
                    left++
                }
                for left < right && nums[right] == nums[right-1]{
                    right--
                }
                left++
                right--
            }
            
        }
    }
    return res
}
```

# 7、无重复字符的最长子串

![image-20260208143437746](C:\Users\Jackson Zhang\AppData\Roaming\Typora\typora-user-images\image-20260208143437746.png)

```go
//滑动窗口
func lengthOfLongestSubstring(s string) int {
    hasMap := make(map[byte]int)
    res := 0
    left := 0
    for right:=0;right<len(s);right++{
        hasMap[s[right]]++
        for hasMap[s[right]] > 1 {
            hasMap[s[left]]--
            left++
        }
        res = max(res,right-left+1)
    }
    return res
}
```

# 8、接雨水

![image-20260208145711410](C:\Users\Jackson Zhang\AppData\Roaming\Typora\typora-user-images\image-20260208145711410.png)

```go
func trap(height []int) int {
    left ,right := 0,len(height)-1
    leftMax,rightMax := 0,0
    ans :=0
    for left < right {
        leftMax = max(leftMax,height[left])
        rightMax = max(rightMax,height[right])
        if height[left] < height[right]{
            ans += leftMax - height[left]
            left++
        }else{
            ans += rightMax - height[right]
            right--
        }
    }
    return ans
}
```

# 9.找到字符串中所有字母异位词

![image-20260209223323206](C:\Users\Jackson Zhang\AppData\Roaming\Typora\typora-user-images\image-20260209223323206.png)

```
func findAnagrams(s string, p string) []int {
	res := make([]int, 0)
	cntP := [26]int{}
	for _, c := range p {
		cntP[c-'a']++
	}

	cntS := [26]int{}
	for right, c := range s {
		cntS[c-'a']++
		left := right - len(p) + 1
		if left < 0 {
			continue
		}
		if cntS == cntP {
			res = append(res, left)
		}
		cntS[s[left]-'a']--
	}
	return res
}

```

# 10、和为k的子数组

![image-20260209225522740](C:\Users\Jackson Zhang\AppData\Roaming\Typora\typora-user-images\image-20260209225522740.png)

```
//暴力求解
func subarraySum(nums []int, k int) int {
    count := 0
    for start := 0; start < len(nums); start++ {
        sum := 0
        for end := start; end >= 0; end-- {
            sum += nums[end]
            if sum == k {
                count++
            }
        }
    }
    return count
}
//前缀和
func subarraySum(nums []int, k int) (ans int) {
    cnt := make(map[int]int, len(nums)+1) // 预分配空间
    cnt[0] = 1 // s[0]=0 单独统计
    s := 0
    for _, x := range nums {
        s += x
        ans += cnt[s-k]
        cnt[s]++
    }
    return
}

```

11、滑动窗口最大值

![image-20260209233116199](C:\Users\Jackson Zhang\AppData\Roaming\Typora\typora-user-images\image-20260209233116199.png)

```
func maxSlidingWindow(nums []int, k int) []int {
    ans := make([]int,len(nums)-k+1)
    q := []int{}
    for i,x := range nums{
        for len(q) > 0 && nums[q[len(q)-1]] <=x {
            q = q[:len(q)-1]
        }
        q = append(q,i)

        left := i -k + 1
        if q[0] < left{
            q = q[1:]
        }

        if left >= 0{
            ans[left] = nums[q[0]]
        }
    }
    return ans
}
```

12、最小覆盖子串

13、最大子数组和

14、合并区间

15、轮转数组

16、除了自身以外数组的乘积

17、缺失的第一个正数

# 18、矩阵置零

![image-20260211211935441](C:\Users\Jackson Zhang\AppData\Roaming\Typora\typora-user-images\image-20260211211935441.png)

```go
func setZeroes(matrix [][]int)  {
    row := make(map[int]bool,len(matrix))
    col := make(map[int]bool,len(matrix[0]))
    for i,r := range matrix{
        for j,c := range r{
            if c == 0{
                row[i] = true
                col[j] = true
            }
        }
    }

    for i:=0;i<len(matrix);i++{
        for j:=0;j<len(matrix[0]);j++{
            if row[i] == true ||col[j] == true{
                matrix[i][j] = 0
            }
        }
    }
}
```

# 19、螺旋矩阵

![image-20260211215207521](C:\Users\Jackson Zhang\AppData\Roaming\Typora\typora-user-images\image-20260211215207521.png)

```
func spiralOrder(matrix [][]int) []int {
	if len(matrix) == 0 || len(matrix[0]) == 0 {
		return []int{}
	}
	m := len(matrix)
	n := len(matrix[0])
	res := make([]int, m*n)
	dir := [][]int{[]int{0, 1}, []int{1, 0}, []int{0, -1}, []int{-1, 0}}
	directionIndex := 0
	row, column := 0, 0
	visited := make([][]bool, m)
	for i := 0; i < m; i++ {
		visited[i] = make([]bool, n)
	}
	for i := 0; i < m*n; i++ {
		res[i] = matrix[row][column]
        visited[row][column] = true
		nextRow := row + dir[directionIndex][0]
		nextCol := column + dir[directionIndex][1]
		if nextRow < 0 || nextRow >= m || nextCol < 0 || nextCol >= n || visited[nextRow][nextCol] {
			directionIndex = (directionIndex + 1) % 4
		}
		row += dir[directionIndex][0]
		column += dir[directionIndex][1]
	}
	return res

}
```

# 20、旋转图像

![image-20260211215241595](C:\Users\Jackson Zhang\AppData\Roaming\Typora\typora-user-images\image-20260211215241595.png)

```
func rotate(matrix [][]int)  {
    n := len(matrix)
    //转置，
    for i:=0;i<n;i++{
        for j:=0;j<i;j++{
            matrix[i][j],matrix[j][i] = matrix[j][i],matrix[i][j]
        }
    }

    //反转
    for i:=0;i<n;i++{
        slices.Reverse(matrix[i])
    }

}
疑问：这里为什么是j<i,而不是j<n?
j<i表示只遍历下三角，只会交换一次。
j<n：
当 i=0, j=1：交换 (0,1) 和 (1,0)
当 i=1, j=0：又交换 (1,0) 和 (0,1) → 换回来了！
最终矩阵不变（因为每对元素被交换了两次）
```

# 21、搜索二维矩阵

![image-20260211222507857](C:\Users\Jackson Zhang\AppData\Roaming\Typora\typora-user-images\image-20260211222507857.png)

```
func searchMatrix(matrix [][]int, target int) bool {
    for _, row := range matrix {
        for _, v := range row {
            if v == target {
                return true
            }
        }
    }
    return false
}
```

# 22、相交链表

![image-20260211224012451](C:\Users\Jackson Zhang\AppData\Roaming\Typora\typora-user-images\image-20260211224012451.png)

```
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func getIntersectionNode(headA, headB *ListNode) *ListNode {
    if headA == nil || headB == nil{
        return nil
    }
    
    hasMap := make(map[*ListNode]bool)
    cur := headA
    for cur != nil  {
        hasMap[cur] = true
        cur = cur.Next
    }
    cur = headB
    for cur != nil{
        if hasMap[cur]{
            return cur
        }
        cur = cur.Next
    }
    return nil

}
```

# 23、反转链表

![image-20260212112013654](C:\Users\Jackson Zhang\AppData\Roaming\Typora\typora-user-images\image-20260212112013654.png)

```
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func reverseList(head *ListNode) *ListNode {
    if head == nil{
        return nil
    }
	var left *ListNode
	for head != nil {
		next := head.Next
        head.Next = left
        left = head
        head = next
	}
	return left
}
```

# 24、回文链表

![image-20260212113910785](C:\Users\Jackson Zhang\AppData\Roaming\Typora\typora-user-images\image-20260212113910785.png)

```
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
 1、使用数组
func isPalindrome(head *ListNode) bool {
    res := make([]int,0)
    for head != nil{
        res = append(res,head.Val)
        head = head.Next
    }
    
    i:=0
    j:=len(res)-1
    for i<j{
        if res[i] != res[j]{
            return false
        }
        i++
        j--
    }
    return true

}
2、找到中间结点，反转链表
// 876. 链表的中间结点
func middleNode(head *ListNode) *ListNode {
    slow, fast := head, head
    for fast != nil && fast.Next != nil {
        slow = slow.Next
        fast = fast.Next.Next
    }
    return slow
}

// 206. 反转链表
func reverseList(head *ListNode) *ListNode {
    var pre, cur *ListNode = nil, head
    for cur != nil {
        nxt := cur.Next
        cur.Next = pre
        pre = cur
        cur = nxt
    }
    return pre
}

func isPalindrome(head *ListNode) bool {
    mid := middleNode(head)
    head2 := reverseList(mid)
    for head2 != nil {
        if head.Val != head2.Val { // 不是回文链表
            return false
        }
        head = head.Next
        head2 = head2.Next
    }
    return true
}

```

# 25、环形链表

![image-20260212114938411](C:\Users\Jackson Zhang\AppData\Roaming\Typora\typora-user-images\image-20260212114938411.png)

```
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func hasCycle(head *ListNode) bool {
    slow, fast := head, head // 乌龟和兔子同时从起点出发
    for fast != nil && fast.Next != nil {
        slow = slow.Next // 乌龟走一步
        fast = fast.Next.Next // 兔子走两步
        if fast == slow { // 兔子追上乌龟（套圈），说明有环
            return true
        }
    }
    return false // 访问到了链表末尾，无环
}
```

# 26、环形列表2

![image-20260212120659011](C:\Users\Jackson Zhang\AppData\Roaming\Typora\typora-user-images\image-20260212120659011.png)

```
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func detectCycle(head *ListNode) *ListNode {
	hasMap := make(map[*ListNode]struct{})
    cur := head
    for cur != nil{
        if _,ok:=hasMap[cur];ok{
            return cur
        }
        hasMap[cur] = struct{}{}
        cur = cur.Next
    }
    return nil
}
```

27、合并两个有序链表

![image-20260213105141264](C:\Users\Jackson Zhang\AppData\Roaming\Typora\typora-user-images\image-20260213105141264.png)

```
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
    var dammy = &ListNode{}
    cur := dammy
    list1P := list1
    list2P := list2
    for list1P != nil && list2P != nil{
        if list1P.Val <= list2P.Val{
            cur.Next = list1P
            cur = cur.Next
            list1P = list1P.Next
        }else{
            cur.Next = list2P
            cur = cur.Next
            list2P = list2P.Next
        }
    }
    if list1P == nil{
        cur.Next = list2P
    }
    if list2P == nil{
        cur.Next = list1P
    }
    return dammy.Next
}
```

28、两数相加

![image-20260213111449184](C:\Users\Jackson Zhang\AppData\Roaming\Typora\typora-user-images\image-20260213111449184.png)

```
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
    var head *ListNode
    var tail = head
    carry := 0
    for l1 != nil || l2 != nil{
        n1,n2:=0,0
        if l1 != nil{
            n1 = l1.Val
            l1 = l1.Next
        }
        if l2 != nil{
            n2 = l2.Val
            l2 = l2.Next
        }
        sum := n1 + n2 +carry
        sum,carry = sum%10,sum/10
        if head == nil{
            head = &ListNode{Val:sum}
            tail = head
        }else{
            tail.Next = &ListNode{Val:sum}
            tail = tail.Next
        }
    }
    if carry > 0{
        tail.Next = &ListNode{Val:carry}
    } 
    return head
}
```

29、删除链表的倒数第N个结点

![image-20260213113120921](C:\Users\Jackson Zhang\AppData\Roaming\Typora\typora-user-images\image-20260213113120921.png)

```
/**
 * Definition for singly-linked list.
 * type ListNode struct {
 *     Val int
 *     Next *ListNode
 * }
 */
func removeNthFromEnd(head *ListNode, n int) *ListNode {
    len := 0
    for cur:=head;cur!=nil;cur=cur.Next{
        len++
    }
    var hammy = &ListNode{Next:head}
    cur := hammy
    for i:=0;i<len-n;i++{
        cur =cur.Next
    }
    cur.Next = cur.Next.Next
    return hammy.Next
}
```

30、两两交换链表中的节点

![image-20260213123039588](C:\Users\Jackson Zhang\AppData\Roaming\Typora\typora-user-images\image-20260213123039588.png)

```
func swapPairs(head *ListNode) *ListNode {
    if head == nil || head.Next == nil{
        return head
    }
    cur := head
    dammy := &ListNode{Next:head}
    pre := dammy
    for cur != nil && cur.Next != nil{
        pre.Next = cur.Next
        cur.Next = cur.Next.Next
        pre.Next.Next = cur
        pre = cur 
        cur = cur.Next
    }
    return dammy.Next
}
```

