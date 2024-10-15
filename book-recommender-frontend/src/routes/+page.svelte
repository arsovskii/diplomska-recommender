<script>
	import Book from '../components/book.svelte';
	import Bookcarousel from '../components/bookCarousel.svelte';
	import Header from '../components/header.svelte';

	import { onMount } from 'svelte';

	let showFirstTime = false;
	let favoriteGenre = '';

	onMount(() => {
		// Check if the user already has a favorite genre stored
		const storedGenre = localStorage.getItem('favoriteGenre');
		if (!storedGenre) {
			// Show the modal if no favorite genre is stored
			showFirstTime = true;
		} else {
			favoriteGenre = storedGenre;
		}
	});

	function saveGenre() {
		// Save the selected genre to localStorage
		localStorage.setItem('favoriteGenre', favoriteGenre);
		showFirstTime = false;
	}
</script>

{#if showFirstTime}
	da da eve modal
{/if}
<div class="w-3/4 mx-auto my-5 relative">
	<Bookcarousel></Bookcarousel>
</div>
