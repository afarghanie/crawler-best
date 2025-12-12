import re

def test_regex():
    pattern = r"<a[^>]+(?:rel=['\"][^'\"]*\bnext\b[^'\"]*['\"])[^>]*href=['\"]([^'\"]+)['\"]|<a[^>]+href=['\"]([^'\"]+)['\"][^>]*(?:rel=['\"][^'\"]*\bnext\b[^'\"]*['\"])"
    
    cases = [
        ('<a href="https://example.com/2" rel="next">Next</a>', "https://example.com/2"),
        ('<a rel="next" href="/page/2" class="btn">Next</a>', "/page/2"),
        ('<a class="x" rel="next" id="y" href="foo"></a>', "foo"),
        ('<a href="bar" id="z" rel="next"></a>', "bar"),
        ('<a rel="prev" href="wrong"></a>', None),
        ('<a href="wrong">Next</a>', None),
        ('<a href="ok" rel="next nofollow">Next</a>', "ok"),
        ('<a rel="nofollow next" href="ok2">Next</a>', "ok2"),
    ]
    
    for html, expected in cases:
        matches = re.findall(pattern, html, re.IGNORECASE)
        res = None
        for m in matches:
            res = m[0] or m[1]
            break
            
        if res == expected:
            print(f"PASS: {html}")
        else:
            print(f"FAIL: {html} -> Expected {expected}, Got {res}")

if __name__ == "__main__":
    test_regex()
