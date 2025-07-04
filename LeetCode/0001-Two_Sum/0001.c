/**
 * Note: The returned array must be malloced, assume caller calls free().
 */
int* twoSum(int* nums, int numsSize, int target, int* returnSize) {
    *returnSize=2;
    int *ans=(int *)malloc(sizeof(int)*2);
    for(int i=0;i<numsSize;i++){
        for(int j=0;j<numsSize;j++){
            if(i!=j){
                if(target==nums[i]+nums[j]){
                    ans[0]=i;
                    ans[1]=j;
                }
            }
        }
    }
    return ans;
}
