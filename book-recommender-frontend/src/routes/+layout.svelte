<script>
	import '../app.css';
	import '@fortawesome/fontawesome-free/css/all.min.css';
	import Header from '../components/header.svelte';

	import { beforeNavigate, afterNavigate } from '$app/navigation';

	import { fade } from 'svelte/transition';
	import { cubicIn, cubicOut } from 'svelte/easing';
	import Loader from '../components/loader.svelte';
	import Searchmodal from '../components/searchmodal.svelte';

	let isLoading = false;

	beforeNavigate(({ to }) => {
		if (to && to.route && to.route.id) {
			isLoading = true;
		}
	});
	afterNavigate(() => (isLoading = false));

	export let data;
</script>

<Header></Header>

{#if isLoading}
	<div
		in:fade={{ easing: cubicOut, duration: 300, delay: 400 }}
		out:fade={{ easing: cubicIn, duration: 300 }}
		class="w-full h-full relative"
	>
		<Loader />
	</div>
{/if}
{#key data.pathname}
	<div
		in:fade={{ easing: cubicOut, duration: 300, delay: 400 }}
		out:fade={{ easing: cubicIn, duration: 300 }}
	>
		<slot />
	</div>
{/key}

<Searchmodal></Searchmodal>
