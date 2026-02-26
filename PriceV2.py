import pandas as pd
from tqdm import tqdm

# ---------------------------
#  Configuration
# ---------------------------
SHIPPING_COST = 50
MULTIPLIER = 2.3
HANDLING_FEE = 40
FIXED_HANDLING = 60

# eBay Gold sale discount rate (60% off strike price → lands at New_Price)
EBAY_GOLD_SALE_DISCOUNT = 0.60
OVERPRICED_THRESHOLD = 500

# ---------------------------
#  Marketplace Fee Setup
# ---------------------------
FEE_TABLE = {
    'Amazon': 0.1500,
    'Walmart': 0.1559,
    'eBay Silver': 0.1600,
    'eBay Gold': 0.1600,
    'Etsy': 0.1050,
    'Shopify': 0.0310,
}

print('Choose marketplace:')
print(*[f'• {m}' for m in FEE_TABLE], sep='\n')
marketplace = input('Marketplace name: ').strip()
fee_rate = FEE_TABLE.get(marketplace)
if fee_rate is None:
    raise ValueError(f"Unknown marketplace '{marketplace}'")

is_ebay_gold = (marketplace == 'eBay Gold')

# ---------------------------
#  Engraving Option
# ---------------------------
include_engraving = input("Include engraving fees (+$30 for SKUs with 'ENG')? (yes/no): ").strip().lower()
add_engraving_fee = include_engraving == 'yes'

# ---------------------------
#  Helper Functions
# ---------------------------
def normalize_style(s: pd.Series) -> pd.Series:
    """Convert style numbers to consistent string format"""
    x = pd.to_numeric(s, errors="coerce")
    x = x.round(0).astype("Int64")
    return x.astype(str)

def is_blank(val) -> bool:
    """Check if value is empty/NA"""
    if val is None or pd.isna(val):
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    return False

def get_batch_factor(batch):
    """Get batch adjustment factor"""
    return {
        'B1': 1.00, 'B2': 1.02, 'Batch2': 1.02,
        'B3': 0.99, 'Batch3': 0.99, 'B4': 1.04,
        'Batch4': 1.04, 'B5': 0.98, 'Batch5': 0.98
    }.get((batch or ''), 1.0)

# ---------------------------
#  Load Data
# ---------------------------
print("\nLoading data files...")

style_df = pd.read_csv('Style-Metalprice(2026).csv').rename(columns=str.strip)
style_df['Style'] = normalize_style(style_df['Style'])
style_df = style_df.dropna(subset=['Style'])

sku_df = pd.read_csv('List.csv', low_memory=False).rename(columns=str.strip)
sku_df['price'] = pd.to_numeric(sku_df['price'], errors='coerce')
sku_df['Style'] = normalize_style(sku_df['Style'])
sku_df = sku_df.dropna(subset=['Style'])

gem_df = pd.read_csv('GemstonePricing(October,2024).csv').rename(columns=str.strip)
gem_df['Gemstone'] = gem_df['Gemstone'].astype(str).str.strip()

# ---------------------------
#  Calculate Pricing
# ---------------------------
def calculate_pricing(row):
    """Calculate complete pricing breakdown with discount scenarios"""
    result = {
        # Cost Components
        'Gold_Cost': 0.0,
        'Gemstone_Cost': 0.0,
        'Shipping_Cost': SHIPPING_COST,
        'Handling_Cost': HANDLING_FEE,
        'Engraving_Cost': 0.0,
        'Fixed_ACOG': FIXED_HANDLING,
        
        # Pricing
        'Old_Price': 0.0,
        'New_Price': 0.0,
        'Price_Difference_$': 0.0,
        'Price_Difference_%': 0.0,
        'Gold_%_of_Price': 0.0,
        
        # eBay Gold Strike-Through Price
        # Strike_Price is the inflated "before sale" price.
        # Applying the 60% discount to Strike_Price brings the customer
        # back to exactly New_Price (the true calculated selling price).
        # Formula: Strike_Price = New_Price / (1 - 0.60) = New_Price / 0.40
        'Strike_Price': 0.0,          # listed / crossed-out price on eBay Gold
        'Sale_Discount_%': 0.0,       # always 60 for eBay Gold, 0 otherwise
        'Sale_Price': 0.0,            # price customer actually pays (= New_Price)

        # Overpriced flag (Sale_Price > $500)
        'Overpriced': '',
        
        # Full Price Metrics
        'Marketplace_Fee': 0.0,
        'Net_Revenue': 0.0,
        'Total_ACOG': 0.0,
        'Profit': 0.0,
        'Profit_Margin_%': 0.0,
        
        # 5% Discount
        'Price_at_5%_Discount': 0.0,
        'Profit_at_5%_Discount': 0.0,
        'Margin_at_5%_Discount': 0.0,
        
        # 10% Discount
        'Price_at_10%_Discount': 0.0,
        'Profit_at_10%_Discount': 0.0,
        'Margin_at_10%_Discount': 0.0,
        
        # 15% Discount
        'Price_at_15%_Discount': 0.0,
        'Profit_at_15%_Discount': 0.0,
        'Margin_at_15%_Discount': 0.0,
        
        'Error': None
    }
    
    try:
        # Capture old price
        old_price = row.get('price', 0)
        if not is_blank(old_price):
            result['Old_Price'] = float(old_price)
        
        # Get style data
        style_key = str(row.get('Style', '')).strip()
        style_row = style_df.loc[style_df['Style'] == style_key]
        
        if style_row.empty:
            result['Error'] = f"Style {style_key} not found"
            return result
        
        style_data = style_row.iloc[0]
        
        # STEP 1: Get gold/metal cost
        metal_purity = str(row.get('Metal Purity', '')).upper()
        metal_col = '18k price' if '18K' in metal_purity else '14k price'
        metal_price = style_data.get(metal_col, 0)
        
        if not is_blank(metal_price):
            result['Gold_Cost'] = float(metal_price)
        
        # STEP 2: Get gemstone cost
        gem_key = str(row.get('Gemstone', '')).strip()
        gem_row = gem_df.loc[gem_df['Gemstone'] == gem_key]
        
        if not gem_row.empty:
            gem_unit_price = float(gem_row['Price'].iloc[0])
            main_weight = style_data.get('Main Stone Weight', 0)
            gem_factor = style_data.get('Gem Factor', 0)
            
            if not is_blank(main_weight) and not is_blank(gem_factor):
                result['Gemstone_Cost'] = float(main_weight) * gem_unit_price * float(gem_factor)
        
        # STEP 3: Calculate base price
        # Formula: ((Gold + Gem + Shipping) * 2.3) + Handling
        subtotal = result['Gold_Cost'] + result['Gemstone_Cost'] + SHIPPING_COST
        base_price = (subtotal * MULTIPLIER) + HANDLING_FEE
        
        # STEP 4: Apply batch factor
        batch_factor = get_batch_factor(row.get('Batch', ''))
        result['New_Price'] = base_price * batch_factor
        
        # STEP 5: Add engraving if applicable
        seller_sku = str(row.get('seller-sku', '')).upper()
        if add_engraving_fee and 'ENG' in seller_sku:
            result['Engraving_Cost'] = 30
            result['New_Price'] += 30
        
        # STEP 6: eBay Gold — calculate strike-through price
        # The customer-facing "sale price" equals New_Price (what they actually pay).
        # The strike-through price is inflated so that 60% off lands exactly at New_Price.
        #   Strike_Price × (1 - 0.60) = New_Price
        #   Strike_Price = New_Price / 0.40
        if is_ebay_gold:
            result['Strike_Price'] = result['New_Price'] / (1 - EBAY_GOLD_SALE_DISCOUNT)
            result['Sale_Discount_%'] = EBAY_GOLD_SALE_DISCOUNT * 100   # 60
            result['Sale_Price'] = result['New_Price']                   # what customer pays
        else:
            result['Strike_Price'] = 0.0
            result['Sale_Discount_%'] = 0.0
            result['Sale_Price'] = result['New_Price']

        # STEP 7: Overpriced flag — based on the price the customer actually pays
        result['Overpriced'] = 'Overpriced' if result['Sale_Price'] > OVERPRICED_THRESHOLD else ''

        # STEP 8: Calculate price difference vs old price
        if result['Old_Price'] > 0:
            result['Price_Difference_$'] = result['New_Price'] - result['Old_Price']
            result['Price_Difference_%'] = (result['Price_Difference_$'] / result['Old_Price']) * 100
        
        # STEP 9: Calculate gold percentage of final price
        if result['New_Price'] > 0:
            result['Gold_%_of_Price'] = (result['Gold_Cost'] / result['New_Price']) * 100
        
        # STEP 10: Calculate ACOG (All Cost of Goods)
        result['Total_ACOG'] = (
            result['Gold_Cost'] + 
            result['Gemstone_Cost'] + 
            result['Shipping_Cost'] + 
            result['Fixed_ACOG'] + 
            result['Engraving_Cost']
        )
        
        # STEP 11: New Price Profitability (revenue based on what customer pays = Sale_Price = New_Price)
        result['Marketplace_Fee'] = result['Sale_Price'] * fee_rate
        result['Net_Revenue'] = result['Sale_Price'] - result['Marketplace_Fee']
        result['Profit'] = result['Net_Revenue'] - result['Total_ACOG']
        
        if result['Net_Revenue'] > 0:
            result['Profit_Margin_%'] = (result['Profit'] / result['Net_Revenue']) * 100
        
        # STEP 12: Calculate discount scenarios (applied on top of New_Price / Sale_Price)
        for discount_pct in [5, 10, 15]:
            discount_price = result['Sale_Price'] * (1 - discount_pct/100)
            discount_fee = discount_price * fee_rate
            discount_net = discount_price - discount_fee
            discount_profit = discount_net - result['Total_ACOG']
            discount_margin = (discount_profit / discount_net * 100) if discount_net > 0 else 0
            
            result[f'Price_at_{discount_pct}%_Discount'] = discount_price
            result[f'Profit_at_{discount_pct}%_Discount'] = discount_profit
            result[f'Margin_at_{discount_pct}%_Discount'] = discount_margin
        
    except Exception as e:
        result['Error'] = str(e)
    
    return result

# ---------------------------
#  Process All SKUs
# ---------------------------
print("Calculating pricing...")
results = []

for _, row in tqdm(sku_df.iterrows(), total=len(sku_df), desc='Processing'):
    results.append(calculate_pricing(row))

results_df = pd.DataFrame(results)
sku_df = pd.concat([sku_df, results_df], axis=1)

# ---------------------------
#  Create Clean Output
# ---------------------------
output_cols = [
    # Identifiers
    'seller-sku',
    'asin1',
    'Style',
    'Gemstone',
    'Metal Purity',
    'Batch',
    
    # === PRICE COMPARISON ===
    'Old_Price',
    'New_Price',
    'Price_Difference_$',
    'Price_Difference_%',
    
    # === EBAY GOLD SALE PRICING ===
    # (Strike_Price and Sale_Discount_% are populated only for eBay Gold;
    #  zero for all other marketplaces)
    'Strike_Price',
    'Sale_Discount_%',
    'Sale_Price',

    # === OVERPRICED FLAG ===
    'Overpriced',
    
    # === COST BREAKDOWN ===
    'Gold_Cost',
    'Gemstone_Cost',
    'Shipping_Cost',
    'Handling_Cost',
    'Engraving_Cost',
    'Fixed_ACOG',
    'Total_ACOG',
    
    # === PRICING METRICS ===
    'Gold_%_of_Price',
    
    # === NEW PRICE PROFITABILITY ===
    'Marketplace_Fee',
    'Net_Revenue',
    'Profit',
    'Profit_Margin_%',
    
    # === 5% DISCOUNT ===
    'Price_at_5%_Discount',
    'Profit_at_5%_Discount',
    'Margin_at_5%_Discount',
    
    # === 10% DISCOUNT ===
    'Price_at_10%_Discount',
    'Profit_at_10%_Discount',
    'Margin_at_10%_Discount',
    
    # === 15% DISCOUNT ===
    'Price_at_15%_Discount',
    'Profit_at_15%_Discount',
    'Margin_at_15%_Discount',
    
    'Error'
]

# Round numeric columns
numeric_cols = [
    'Old_Price', 'New_Price', 'Price_Difference_$', 'Price_Difference_%',
    'Strike_Price', 'Sale_Discount_%', 'Sale_Price',
    'Gold_Cost', 'Gemstone_Cost', 'Shipping_Cost', 'Handling_Cost', 
    'Engraving_Cost', 'Fixed_ACOG', 'Total_ACOG',
    'Gold_%_of_Price',
    'Marketplace_Fee', 'Net_Revenue', 'Profit', 'Profit_Margin_%',
    'Price_at_5%_Discount', 'Profit_at_5%_Discount', 'Margin_at_5%_Discount',
    'Price_at_10%_Discount', 'Profit_at_10%_Discount', 'Margin_at_10%_Discount',
    'Price_at_15%_Discount', 'Profit_at_15%_Discount', 'Margin_at_15%_Discount'
]

for col in numeric_cols:
    if col in sku_df.columns:
        sku_df[col] = pd.to_numeric(sku_df[col], errors='coerce').round(2)

output_df = sku_df[output_cols].copy()
output_df = output_df.rename(columns={
    'seller-sku': 'SKU',
    'asin1': 'ASIN'
})

# ---------------------------
#  Export Results
# ---------------------------
output_filename = f'Simplified_Pricing_{marketplace.replace(" ", "_")}.csv'
output_df.to_csv(output_filename, index=False)

# ---------------------------
#  Summary Statistics
# ---------------------------
print("\n" + "="*80)
print(f"SIMPLIFIED PRICING ANALYSIS - {marketplace}")
print("="*80)

total_skus = len(output_df)
skus_with_errors = output_df['Error'].notna().sum()

print(f"\nTotal SKUs Processed: {total_skus}")
print(f"SKUs with Errors: {skus_with_errors}")

print(f"\n{'PRICE COMPARISON':-^80}")
print(f"Average Old Price:      ${output_df['Old_Price'].mean():>10,.2f}")
print(f"Average New Price:      ${output_df['New_Price'].mean():>10,.2f}")
print(f"Average Difference:     ${output_df['Price_Difference_$'].mean():>10,.2f}")
print(f"Average % Change:       {output_df['Price_Difference_%'].mean():>10.2f}%")
print(f"\nPrice Increases:        {(output_df['Price_Difference_$'] > 0).sum():>10,} SKUs")
print(f"Price Decreases:        {(output_df['Price_Difference_$'] < 0).sum():>10,} SKUs")
print(f"No Change:              {(output_df['Price_Difference_$'] == 0).sum():>10,} SKUs")

# eBay Gold specific summary
if is_ebay_gold:
    overpriced_count = (output_df['Overpriced'] == 'Overpriced').sum()
    print(f"\n{'EBAY GOLD SALE PRICING':-^80}")
    print(f"Sale Discount:          {EBAY_GOLD_SALE_DISCOUNT*100:.0f}% off Strike Price")
    print(f"Average Strike Price:   ${output_df['Strike_Price'].mean():>10,.2f}  (listed / crossed-out)")
    print(f"Average Sale Price:     ${output_df['Sale_Price'].mean():>10,.2f}  (customer pays = New Price)")
    print(f"\nOverpriced (Sale Price > ${OVERPRICED_THRESHOLD}): {overpriced_count:>6,} SKUs")

print(f"\n{'AVERAGE COST BREAKDOWN':-^80}")
print(f"Gold Cost:           ${output_df['Gold_Cost'].mean():>10,.2f}")
print(f"Gemstone Cost:       ${output_df['Gemstone_Cost'].mean():>10,.2f}")
print(f"Shipping Cost:       ${output_df['Shipping_Cost'].mean():>10,.2f}")
print(f"Handling Cost:       ${output_df['Handling_Cost'].mean():>10,.2f}")
print(f"Engraving Cost:      ${output_df['Engraving_Cost'].mean():>10,.2f}")
print(f"Fixed ACOG:          ${output_df['Fixed_ACOG'].mean():>10,.2f}")
print(f"{'─'*80}")
print(f"Total ACOG:          ${output_df['Total_ACOG'].mean():>10,.2f}")

print(f"\n{'PRICING & PROFITABILITY':-^80}")
print(f"New Price:           ${output_df['New_Price'].mean():>10,.2f}")
print(f"Gold % of Price:     {output_df['Gold_%_of_Price'].mean():>10.2f}%")
print(f"Marketplace Fee:     ${output_df['Marketplace_Fee'].mean():>10,.2f} ({fee_rate*100:.2f}%)")
print(f"Net Revenue:         ${output_df['Net_Revenue'].mean():>10,.2f}")
print(f"Profit:              ${output_df['Profit'].mean():>10,.2f}")
print(f"Profit Margin:       {output_df['Profit_Margin_%'].mean():>10.2f}%")

print(f"\n{'DISCOUNT SCENARIOS':-^80}")
print(f"{'Discount':<12} {'Price':<12} {'Profit':<12} {'Margin':<12}")
print(f"{'─'*80}")
print(f"{'5%':<12} ${output_df['Price_at_5%_Discount'].mean():<11,.2f} "
      f"${output_df['Profit_at_5%_Discount'].mean():<11,.2f} "
      f"{output_df['Margin_at_5%_Discount'].mean():<11.2f}%")
print(f"{'10%':<12} ${output_df['Price_at_10%_Discount'].mean():<11,.2f} "
      f"${output_df['Profit_at_10%_Discount'].mean():<11,.2f} "
      f"{output_df['Margin_at_10%_Discount'].mean():<11.2f}%")
print(f"{'15%':<12} ${output_df['Price_at_15%_Discount'].mean():<11,.2f} "
      f"${output_df['Profit_at_15%_Discount'].mean():<11,.2f} "
      f"{output_df['Margin_at_15%_Discount'].mean():<11.2f}%")

print(f"\n{'MARKETPLACE INFO':-^80}")
print(f"Marketplace: {marketplace}")
print(f"Marketplace Fee: {fee_rate*100:.2f}%")
print(f"Engraving Fees Included: {'Yes' if add_engraving_fee else 'No'}")

print(f"\n✅ Results saved to: {output_filename}")
print("="*80)

# ---------------------------
#  Sample Product Display
# ---------------------------
print(f"\n{'SAMPLE PRODUCT BREAKDOWN':-^80}")
sample = output_df.iloc[0]
print(f"\nSKU: {sample['SKU']}")
print(f"Style: {sample['Style']} | {sample['Metal Purity']} | {sample['Gemstone']}")

print(f"\n{'PRICE COMPARISON':─^40}")
print(f"  Old Price:         ${sample['Old_Price']:>10,.2f}")
print(f"  New Price:         ${sample['New_Price']:>10,.2f}")
print(f"  Difference:        ${sample['Price_Difference_$']:>10,.2f}")
print(f"  % Change:          {sample['Price_Difference_%']:>10.2f}%")

if is_ebay_gold:
    print(f"\n{'EBAY GOLD SALE':─^40}")
    print(f"  Strike Price:      ${sample['Strike_Price']:>10,.2f}  ← listed (crossed-out)")
    print(f"  Sale Discount:     {sample['Sale_Discount_%']:>10.0f}%  off strike price")
    print(f"  Sale Price:        ${sample['Sale_Price']:>10,.2f}  ← customer pays")
    print(f"  Overpriced:        {'Yes' if sample['Overpriced'] == 'Overpriced' else 'No':>10}")

print(f"\n{'COST COMPONENTS':─^40}")
print(f"  Gold:              ${sample['Gold_Cost']:>10,.2f}")
print(f"  Gemstone:          ${sample['Gemstone_Cost']:>10,.2f}")
print(f"  Shipping:          ${sample['Shipping_Cost']:>10,.2f}")
print(f"  Handling:          ${sample['Handling_Cost']:>10,.2f}")
print(f"  Engraving:         ${sample['Engraving_Cost']:>10,.2f}")
print(f"  Fixed ACOG:        ${sample['Fixed_ACOG']:>10,.2f}")
print(f"  {'─'*40}")
print(f"  Total ACOG:        ${sample['Total_ACOG']:>10,.2f}")

print(f"\n{'PROFITABILITY AT NEW PRICE':─^40}")
print(f"  Gold is {sample['Gold_%_of_Price']:.1f}% of price")
print(f"  Marketplace Fee:   ${sample['Marketplace_Fee']:>10,.2f}")
print(f"  Net Revenue:       ${sample['Net_Revenue']:>10,.2f}")
print(f"  Profit:            ${sample['Profit']:>10,.2f}")
print(f"  Margin:            {sample['Profit_Margin_%']:>10.2f}%")

print(f"\n{'AT 5% DISCOUNT':─^40}")
print(f"  Sale Price:        ${sample['Price_at_5%_Discount']:>10,.2f}")
print(f"  Profit:            ${sample['Profit_at_5%_Discount']:>10,.2f}")
print(f"  Margin:            {sample['Margin_at_5%_Discount']:>10.2f}%")

print(f"\n{'AT 10% DISCOUNT':─^40}")
print(f"  Sale Price:        ${sample['Price_at_10%_Discount']:>10,.2f}")
print(f"  Profit:            ${sample['Profit_at_10%_Discount']:>10,.2f}")
print(f"  Margin:            {sample['Margin_at_10%_Discount']:>10.2f}%")

print(f"\n{'AT 15% DISCOUNT':─^40}")
print(f"  Sale Price:        ${sample['Price_at_15%_Discount']:>10,.2f}")
print(f"  Profit:            ${sample['Profit_at_15%_Discount']:>10,.2f}")
print(f"  Margin:            {sample['Margin_at_15%_Discount']:>10.2f}%")
print("="*80)