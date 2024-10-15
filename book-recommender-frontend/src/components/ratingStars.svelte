<script>
	import { onMount } from 'svelte';

	export let id = 0;
	export let rating = 0;

	onMount(() => {
		const finalRating = Math.round(rating * 2);
		console.log(finalRating);
	});
    /*
    <input type="radio" name="rating-{id}" class="mask mask-star bg-secondary" />
	<input type="radio" name="rating-{id}" class="mask mask-star bg-secondary" checked="checked" />
	<input type="radio" name="rating-{id}" class="mask mask-star bg-secondary" />
	<input type="radio" name="rating-{id}" class="mask mask-star bg-secondary" />
	<input type="radio" name="rating-{id}" class="mask mask-star bg-secondary" />
    */
</script>

<div class="rating rating-lg rating-half">
	<input type="radio" name="rating-{id}" class="rating-hidden" />
	<input type="radio" name="rating-{id}" class="mask mask-star-2 mask-half-1 bg-secondary" />
	<input type="radio" name="rating-{id}" class="mask mask-star-2 mask-half-2 bg-secondary" />
	<input type="radio" name="rating-{id}" class="mask mask-star-2 mask-half-1 bg-secondary" />
	<input type="radio" name="rating-{id}" class="mask mask-star-2 mask-half-2 bg-secondary" />
	<input type="radio" name="rating-{id}" class="mask mask-star-2 mask-half-1 bg-secondary" />
	<input type="radio" name="rating-{id}" class="mask mask-star-2 mask-half-2 bg-secondary" />
	<input type="radio" name="rating-{id}" class="mask mask-star-2 mask-half-1 bg-secondary" />
	<input type="radio" name="rating-{id}" class="mask mask-star-2 mask-half-2 bg-secondary" />
	<input type="radio" name="rating-{id}" class="mask mask-star-2 mask-half-1 bg-secondary" />
	<input type="radio" name="rating-{id}" class="mask mask-star-2 mask-half-2 bg-secondary" />
</div>
