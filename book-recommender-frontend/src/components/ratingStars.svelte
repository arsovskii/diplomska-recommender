<script lang="ts">
	import { ratingsStore, updateRating, getRating } from '$lib/stores/stores';
	import { onMount } from 'svelte';

	export let id = 0;
	export let rating = 0;
	export let interactible = true;

	let selectedRating = 0;
	let userRated: Boolean = false;

	let starsContainer: HTMLElement;

	onMount(() => {
		let storeRating = getRating(id);
		console.log(storeRating);
		if (storeRating !== undefined) {
			rating = storeRating;
			userRated = true;
		}
		console.log('rating', rating);
		selectedRating = Math.floor(rating * 2);

		const stars = starsContainer.querySelectorAll('input');
		stars.forEach((star, index) => {
			if (index <= selectedRating) {
				star.checked = true;
			}
			if (!interactible) {
				star.disabled = true;
				star.style.cursor = 'default';
			}
		});
	});

	const updateRatingLocal = (newRating: number) => {
		console.log('oldRating', selectedRating);
		console.log('newRating', newRating);
		selectedRating = newRating;
		userRated = true;

		updateRating(id, newRating);
	};
	/*
    <input type="radio" name="rating-{id}" class="mask mask-star bg-secondary" />
	<input type="radio" name="rating-{id}" class="mask mask-star bg-secondary" checked="checked" />
	<input type="radio" name="rating-{id}" class="mask mask-star bg-secondary" />
	<input type="radio" name="rating-{id}" class="mask mask-star bg-secondary" />
	<input type="radio" name="rating-{id}" class="mask mask-star bg-secondary" />
    */
</script>

<div class="rating rating-lg rating-half" bind:this={starsContainer}>
	<input type="radio" name="rating-{id}" class="rating-hidden" />
	{#each Array(10) as _, index}
		<input
			type="radio"
			name="rating-{id}"
			class={`mask mask-star-2 ${index % 2 === 0 ? 'mask-half-1' : 'mask-half-2'} ${
				userRated ? 'bg-primary' : 'bg-secondary'
			}`}
			on:click={() => updateRatingLocal((index + 1) / 2)}
		/>
	{/each}
</div>
