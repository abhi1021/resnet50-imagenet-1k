#!/usr/bin/env bash

# =====================================================
# Check Spot prices across AWS regions for a given instance type
# and list the Top 5 cheapest regions + AZs
# =====================================================

INSTANCE_TYPE="g5.2xlarge"
PRODUCT_DESC="Linux/UNIX"
LOOKBACK_HOURS=1
TMP_FILE="/tmp/spot_prices.json"

echo "🔍 Checking Spot Prices for $INSTANCE_TYPE across all AWS regions..."
echo ""

# Step 1: Get list of all regions
REGIONS=$(aws ec2 describe-regions --query "Regions[].RegionName" --output text)

# Step 2: Clear temp file
> "$TMP_FILE"

# Step 3: Loop through regions
for REGION in $REGIONS; do
  echo "→ Fetching data from region: $REGION"
  START_TIME=$(date -u -d "-${LOOKBACK_HOURS} hour" +"%Y-%m-%dT%H:%M:%SZ")

  # Get latest spot price for given instance type
  aws ec2 describe-spot-price-history \
    --region "$REGION" \
    --instance-types "$INSTANCE_TYPE" \
    --product-descriptions "$PRODUCT_DESC" \
    --start-time "$START_TIME" \
    --max-results 10 \
    --query "SpotPriceHistory[*].{Region:'$REGION',AZ:AvailabilityZone,Price:SpotPrice,Time:Timestamp}" \
    --output json >> "$TMP_FILE" 2>/dev/null
done

# Step 4: Process and sort results
echo ""
echo "🧮 Sorting results and finding top 5 cheapest..."

jq -s '[.[][]]' "$TMP_FILE" \
  | jq -r '.[] | [.Region, .AZ, .Price, .Time] | @tsv' \
  | sort -k3 -n \
  | awk 'BEGIN { printf "%-15s %-15s %-10s %-25s\n", "Region", "AZ", "Price($/hr)", "Timestamp"
                  print "-------------------------------------------------------------------" }
         { printf "%-15s %-15s %-10s %-25s\n", $1, $2, $3, $4 }' \
  | head -n 6

echo ""
echo "✅ Done."
