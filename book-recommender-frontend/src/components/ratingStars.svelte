<script lang="ts">
	import { onMount } from 'svelte';

	export let id = 0;
	export let rating = 0;

	let selectedRating = 0;
	let userRated: Boolean = false;

	let starsContainer: HTMLElement;

	onMount(() => {
		selectedRating = Math.floor(rating * 2);
		
		const stars = starsContainer.querySelectorAll('input');
		stars.forEach((star, index) => {
			if (index <= selectedRating) {
				star.checked = true;
			}
		});
	});

	const updateRating = (newRating: number) => {
		console.log("oldRating", selectedRating);
		console.log('newRating', newRating);
		selectedRating = newRating;
		userRated = true;
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
			class={`mask mask-star-2 mask-half-${index % 2 === 0 ? 1 : 2} ${
				userRated ? 'bg-primary' : 'bg-secondary'
			}`}
			on:click={() => updateRating(index)}
			
		/>
	{/each}
</div>


