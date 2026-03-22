# Usage Samples

This folder contains copy-paste ready PowerShell samples for both script versions.

Use `v1` when you already know the product info and want fast generation.
Use `v2` when you want the script to inspect the Naver store page first.

## V1 samples

```powershell
.\samples\beauty_sample.ps1
.\samples\health_food_sample.ps1
```

## V2 samples

```powershell
.\samples\living_sample.ps1
.\samples\digital_sample.ps1
```

## Quick templates

### V1

```powershell
.\venv\Scripts\python.exe .\naver_affiliate_blog.py `
  --product-name "Your product name" `
  --product-url "https://example.com/item" `
  --product-category auto `
  --target-segment "Optional target segment" `
  --banned-expressions best guaranteed cure `
  --output-format md
```

### V2

```powershell
.\venv\Scripts\python.exe .\naver_affiliate_blog_v2.py `
  --product-url "https://smartstore.naver.com/your-store/products/1234567890" `
  --product-category living `
  --target-segment "Optional target segment" `
  --banned-expressions best guaranteed cure `
  --output-format md
```
