
```cpp
#include <iostream>
#include <bits/stdc++.h>
#include <conio.h>
#include <sstream>
using namespace std;

string data[52] = {".-", "-...", "-.-.", "-..", ".", "..-.","--.",
                "....", "..", ".---", "-.-", ".-..","--", "-.",
                 "---", ".--.", "--.-", ".-.", "...", "-","..-",
                 "...-", ".--", "-..-", "-.--", "--..",
                 "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P",
                 "Q","R","S","T","U","V","W","X","Y","Z",};
int main()
{
    string s;
    int i;
    while(getline(cin,s))
    {
        istringstream ss(s);
        string t,ans;
        while(ss >> t)
        {
            for(i=0 ; i<52 ; i++)
            {
                if (i<26 && data[i] == t)
                {
                    ans = data[i+26];
                    t.clear();
                    break;
                }
                if(i>=26 && i<52 && data[i] == t)
                {
                    ans = data[i-26];
                    t.clear();
                    break;
                }
            }
            cout << ans << ' ';
            ans.clear();
        }
    }
}
