<script>
	import Book from '../components/book.svelte';
	import Bookcarousel from '../components/bookCarousel.svelte';
	import FirstTime from '../components/firstTime.svelte';
	import Header from '../components/header.svelte';

	import { onMount } from 'svelte';
	import RecommendationButton from '../components/recommendationButton.svelte';

	let showFirstTime = false;
	let favoriteGenre = '';

	onMount(() => {
		// Вчитување на жанровите кои корисникот ги сака
		const storedGenre = localStorage.getItem('favoriteGenres');
		console.log(storedGenre);
		if (!storedGenre) {
			
			showFirstTime = true;
		} else {
			favoriteGenre = storedGenre;
		}
	});
</script>

{#if showFirstTime}
	<FirstTime></FirstTime>
{/if}

<div class="w-3/4 mx-auto my-5 mb-3 relative">
	<Bookcarousel></Bookcarousel>
</div>

<RecommendationButton></RecommendationButton>
