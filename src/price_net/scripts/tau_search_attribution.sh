rm tau_search_attribution.log
for tau in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0; do
    {
        echo "TAU=${tau}"
        uv run evaluate_e2e \
            --config configs/attribution/price_lens_val.yaml \
            --threshold $tau
    } >> tau_search_attribution.log
done
