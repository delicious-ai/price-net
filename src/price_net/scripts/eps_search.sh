for eps in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0; do
    echo "EPSILON=${eps}"
    uv run src/price_net/scripts/eval_heuristic.py \
        --dataset-dir data/price-attribution-scenes \
        --heuristic within_epsilon \
        --results-dir "search/epsilon-${eps}" \
        --epsilon $eps \
        --split val
done
