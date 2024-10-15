<script lang="ts">
	import { Slidy, Core, classNames } from '@slidy/svelte';
	import Book from './book.svelte';
	import { stairs, translate } from '@slidy/animation';
	import { fade } from 'svelte/transition';
	import anime from 'animejs';
	import { onMount } from 'svelte';
	import Loader from './loader.svelte';

	let slides = [
	
	];
	

	let user_id = 1; // Example user ID

	let recommendations: any[] = [];

	const getRecommendations = async () => {
		try {
			const response = await fetch('http://localhost:5000/api/recommend', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify({ user_id })
			});

			const text = await response.text();
			console.log('Raw response:', text);

			if (response.ok) {
				const data = JSON.parse(text);
				recommendations = data.books;
				console.log('Recommendations:', recommendations);
			} else {
				console.error('Failed to fetch recommendations:', response.statusText);
			}
		} catch (error) {
			console.error('Error fetching recommendations:', error);
		}
	};

	// Fetch recommendations on component mount
	// getRecommendations();

	onMount(async () => {
		await getRecommendations();
		hasMounted = true;
		console.log(recommendations)
		slides = recommendations
		slidyMount();
	});

	let hasMounted: Boolean = false;
	let slidyMount = () => {
		if(!hasMounted){
			return;
		}
		let overlay = document.getElementsByClassName('slidy-overlay')[0] as HTMLElement;
		overlay.style.display = 'none';
		
		let allSlidy = document.getElementsByClassName('slidy-slide');
		
		for (let i = 0; i < allSlidy.length; i++) {
			(allSlidy[i] as HTMLElement).style.opacity = '0';
		}

		document.getElementById('slidy-container')!.style.display = 'block';

		anime({
			targets: '.slidy-slide',
			opacity: 1,
			duration: 100,
			delay: anime.stagger(50) // increase delay by 100ms for each elements.
		});
		
	};
</script>


<div transition:fade id="slidy-container" class="transition-all" style="display:none;">
	<Slidy
		{slides}
		let:item
		animation={stairs}
		axis="x"
		snap="center"
		sensitivity="10"
		--slidy-counter-bg="oklch(var(--s))"
		--slidy-arrow-bg="oklch(var(--s))"
		on:mount={slidyMount}
	>
		<figure>
			<Book book={item}></Book>
		</figure>
	</Slidy>
</div>
{#if !hasMounted}
<div class="w-full h-full relative">
	<Loader></Loader>
</div>
{/if}

<style>
	@import url('https://unpkg.com/@slidy/svelte/dist/slidy.css');
</style>
